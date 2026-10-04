# 중복 연산·자료 복제·탐색 병목 개선 계획

작성일: 2026-10-03 UTC. 작성 담당: A(Codex). 확인한 공개 GitHub 작성자:
[daejunnom](https://github.com/daejunnom). 갱신일: 2026-10-03 UTC.
상태: **전체 최적화 A 담당; OPT-01~12 내부 구현·소비자 연결·CPU 검증/반복 진단 완료; 기본 off·GPU/정식 성능·품질 인수 대기**.

이 계획은 현재 소스 조사와 PR #17 재검토에 다음 사용자 지시를 반영한다.

- 기존 계측은 후보 선정의 참고 자료로 사용하고, 새 변경의 성능은 다시 측정한다.
- 자원 경합과 단일 실행의 오염 가능성을 통제하며 여러 차례 비교한다.
- 탐색 의미 또는 전체 실행 흐름이 달라질 수 있는 후보도 실험 코드로 구현한다.
  해당 PR에 실험임을 밝히고 변경에 맞는 테스트 항목을 추가한다.

최신 사용자 배정인 **“이번 최적화는 전부 A, 즉 네가 진행”**를 반영한다. 이 계획의
구현·소비 코드 연결·테스트·계측은 모두 A(Codex)가 진행한다. 범위는 `rz-position`뿐 아니라
`rz-encoding`·`rz-eval`·`rz-search`·`rz-uci`·`rz-runtime`·`rz-telemetry`와 필요한
runner·공통 계약 접점이다. 아래 A/B/C/D/E 표시는 소스의 영역을 설명하며 후속 담당을
나누는 표시가 아니다. 계약 접점이 없으면 내부 구현·CPU/mock 검증을 먼저 진행하고
완료된 선언과 소비 코드를 같은 integration SHA에서 맞춘다. 공통 계약 변경의 영향과
revision은 명시하고 총괄의 통합 리뷰·인수 절차는 유지한다. E/A/S 분류·대국 검증은
[개발 기준](../ENGINEERING-STANDARDS.md), [실험 계약](../EXPERIMENTS.md),
[공통 계약](../CONTRACTS.md)을 따른다. 여기서 E/A/S의 A는 근사·모델 변경 분류이며
작성 담당 A와 다르다. 1~7장의 조사·검증 원칙을 유지하고, 실제 첫 구현·검사·반복 CPU
진단은 8장, 최초 미반영 계획과 실행 순서는 9장, OPT-00 구현·검증은 10장,
OPT-01~12의 최신 구현·검증은 11장에 기록한다. 8·10장의 기존 표본·부정적 결과는
당시 소스의 이력으로 보존한다. 정식 탐색 성능·GPU·대국
강도 인수와 실험의 기본 활성화는 보류 상태다.

## 1. PR #17 재확인과 근거의 범위

| 항목 | 확인 내용 |
|---|---|
| PR | [#17](https://github.com/daejunnom/RoveZero/pull/17), open / draft |
| base | `develop`, `b5ba853585cb5a78f81159f733a86cdfe085936b` |
| 확인한 head | `81059e463af8afc151253b85306fc7c88e5b58bc` |
| PR이 보고한 실제 실행 소스 | `88ea26e714a56af81f43cd1180c31df08fe813e9`; head와 문서 3개만 다름 |
| 계약과 기준 실행 | revision 0.1, W0/S0, FP32/B1, worker 1, ORT intra-thread 1 |
| CI | [37120898712](https://github.com/daejunnom/RoveZero/actions/runs/37120898712)의 Ubuntu·Windows job 및 필수 step 성공 확인 |
| 리뷰 | inline review thread 없음. [총괄 대조 댓글](https://github.com/daejunnom/RoveZero/pull/17#issuecomment-5968915273)의 computed-only·cached 후속 인수 조건 확인 |

문서 변경은 위 develop에서 분기했다. PR #17을 병합된 코드로 간주하지 않으며, 후속 계측은
실제로 선택한 통합 SHA 또는 미병합 head와 그 차이를 실행 manifest에 고정한다.

PR #17은 B 준비, C 인코딩·native worker·Run·출력 변환, D 전달, B 최종 guarded backup의
원래 시각과 identity를 연결한다. optional journal은 metadata 8192개·JSON 8MiB로 제한되며
모델·보드·tensor·device buffer의 소유권을 맡지 않는다. 공통 revision과 최종 guard,
취소·deadline·physical drain·quarantine 계약을 유지하는 계측 연결이다.

[`SourceJournal::record`](https://github.com/daejunnom/RoveZero/blob/81059e463af8afc151253b85306fc7c88e5b58bc/crates/rz-telemetry/src/source.rs#L134)는
공유 `Mutex`를 잠근다. capacity가 제한돼 있어도 시각 취득·lock·기록의 비용은 남는다.
따라서 observer의 기능상 수동성과 시간상 무영향을 동일시하지 않는다. source stamp는
서로 겹칠 수 있고 기록 순서는 사건 순서와 다를 수 있다. 단계별 분위수를 더하지 않는다.

다음 자료의 당시 기능·정확성 근거는 유지하되, 이번 후보의 성능 판정 표본에는 합치지 않는다.

| 참고 자료 | 가설에 주는 정보 | 새 실험에서 다시 확인할 점 |
|---|---|---|
| [PR #16 CPU 최적화 기록](PRE-RUNPOD-OPTIMIZATION.md) | legal 중복 생성·부모 재-export·버리는 undo·불변 profile 해시는 이미 개선됨 | 현재 기준/변경 소스의 실제 비용과 자원 상태 |
| [PR #17 D02 기록](https://github.com/daejunnom/RoveZero/blob/81059e463af8afc151253b85306fc7c88e5b58bc/docs/research/LOCAL-D02-PROFILE.md) | worker 완료→ready P50 약 0.677ms; 같은 game의 두 root 사이 동일 입력 추가 실행 72/258 | 반복 실행의 변동·완료 통지 비용·실제 캐시 적중과 절약량 |
| A의 외부 CPU 진단 probe | 같은 인코딩을 만드는 한 fixture에서 projection의 블록 평균 중앙값이 이력 8개 약 2.34µs, 257개 약 86.99µs | 전체 이력 재순회 비용, 여러 fixture와 독립 실행에서의 재현성 |

마지막 probe는 Rust 1.90.0 release·단일 CPU affinity·9개 교차 블록·블록당 200회 호출이다.
원본은 `${ARTIFACT_ROOT}/perf-audit/probe/paired-results.csv`, `paired-metadata.json`,
`paired-summary.json`, `fixture.json`, `src/main.rs`에 있다. 공유 호스트의 경합을 충분히
관측하지 않았고 한 trace만 사용했으므로 진단 자료다. 이력을 잘라내는 제품 변경의 근거로
사용하지 않는다. PR #17의 258개 요청도 258개의 독립 실행으로 세지 않는다.

PR #17의 off/on 전체 gate에는 bundle copy/hash·load·warm-up·종료·직렬화가 포함된다.
그 시간차를 observer overhead로 사용하지 않는다. ORT Run host interval은 전송·실행·동기화를
포함하며 kernel-only 시간이 아니다. device kernel/transfer와 process peak VRAM은 해당
기록에서 미측정이다. 초기 계획 작성 때는 새 성능 benchmark나 GPU 실행을 수행하지 않았다.

## 2. 남은 병목과 구현 후보

비용 구조는 PR #17 head에서 처음 조사했다. 갱신 때 develop `b5ba853`과 로컬/PR #18
head `15b9d50`에서 남은 구조를 재확인했다. PR #17은 여전히 별도 미병합 실험이다.
실제 구현 전 기준 SHA를 다시 고정한다. 우선순위는 가설이며 성능 개선율을 보장하지 않는다.

| 순서·영역 | 소스와 비용 구조 | 구현 후보 | 현재 상태·후속 단위 | 예상 분류·실험 표시 |
|---|---|---|---|---|
| 1 · A | `position.rs` 반복 횟수·완전성 별도 순회, `outcome.rs`의 intended claim용 임시 child/history/owner | 단일 반복 증거 순회·checked claim preview | #18 opt-in 구현; 인수는 9.3 | E 후보; 동등성 검사 |
| 2 · A | `contracts.rs::state_digest`의 전체 이력 FEN·해시 반복 | FEN 버퍼·현 상태 digest 재사용 | #18 opt-in 구현; 인수는 9.3 | E 후보; 실행 경로 변경 실험 |
| 3 · A/C | `rules_projection.rs::project`의 64칸 재구성·frame별 과거 재순회 | 읽기 전용 비트보드·정확한 반복 정보 | OPT-01 구현; 11장 | E 후보 + 실험; tensor 대조 |
| 4 · C | `input_key`·`prepare`의 projection·encoding·hash·legal index 중복 | 검증된 불변 준비 결과·동일 바이트의 묶음 hash | OPT-02·03 구현; 11장 | E 후보; 준비 공유는 실험 |
| 5 · B | `ContractSearch::pump`의 root부터 재생성·child export 반복 | 유한한 노드 상태 재사용 | OPT-05 구현; 11장 | E 가설 + 실험; 의미 차이는 S 검토 |
| 6 · B | 통계·PUCT 점수 Vec, best move 조회용 전체 outcome·문자열 | 작업 버퍼·streaming argmax·가벼운 조회 | OPT-04 구현; 11장 | E 후보; f64·tie·오류 순서 유지 |
| 7 · C/D/B | computed-only 경로의 동일 입력 재추론 | 같은 game의 root 간 bounded exact raw-eval cache | OPT-06 구현; 11장 | E 가설 + 실험; provenance·receipt 변경 |
| 8 · B/D | Waiting의 1ms sleep·poll | 완료·취소 통지와 deadline 대기 | OPT-07 구현; 11장 | 실험; 고정 방문 trace로 E/S 판정 |
| 9 · A | 모든 pseudo move의 clone/apply/attack, 기하 계산 반복 | 공격 테이블 → pin/check 기반 생성 | OPT-08 구현; 11장 | E 후보 + 실험; movegen 대조 |
| 10 · C/D/B | 매 Run의 staging·출력 복제, B1 단일 진행 요청 | buffer → I/O Binding → 독립 Graph·다중 요청/batch | OPT-09~12 구현; 장치/품질 인수 미실행 | 장치 경계는 E 가설 + 실험; 스케줄 변경은 S |

전체 이력은 persistent prefix를 공유한다. `Arc::clone()`은 일반적으로 전체 자료 복제가
아니므로 실제 allocation/bytes/lock 비용을 확인해 대상을 고른다. 입력 검증·physical lease·
출력 수명에 필요한 소유권은 성능만을 이유로 제거하지 않는다.

현재 state digest는 전체 이력 길이를 앞에 넣고 newest-first canonical FEN을 해시한다.
부모 SHA에 자식만 붙이는 방식으로 같은 digest를 만들 수 없다. O(1) 체인/Merkle codec은
profile·identity 호환성 변경으로 별도 제안하고 기존 golden을 조용히 바꾸지 않는다.
Zobrist도 규칙 상태의 빠른 탐색 키와 전체 모델 이력의 정확한 identity를 구별해 사용한다.

## 3. 자원 경합을 통제하는 반복 계측

### 3.1 실행 전에 잠글 항목

각 비교는 baseline 한 개와 변경 한 개를 사용한다. source SHA·diff, binary/lockfile·
compiler/options·model/export·backend/driver·precision·thread/worker/batch, fixture/seed·
명령 순서·warm-up·캐시 reset/scope, 자원·시간·산출물 상한을 실행 manifest에 기록한다.
관측·통계 방법과 1차 지표, 개선·회귀 허용폭, 오염 판정·재시도·중단 기준도 함께 잠근다.
장비나 영향 입력이 달라지면 새 cohort/run ID를 만들고 이전 결과와 직접 합치지 않는다.

입력군은 startpos, Kiwipete, EP 발견 공격, castling, 네 승격, current/intended claim,
unknown-prefix FEN, 긴 실제 합법 이력, 깊은 탐색, 같은 game의 연속 root를 포함한다.
짧은 이력만 측정하지 않는다. 긴 이력 fixture는 terminal 뒤에 임의 수를 이어 붙이지 않는다.
캐시는 cold miss·같은 game의 cross-root hit·eviction·`ucinewgame` reset을 나누어 측정한다.
성능용 cold/warm 조건을 한 표본 집합으로 합치지 않는다.

### 3.2 실행 자원과 오염 관측

- benchmark가 사용하는 CPU/GPU에서 baseline과 variant를 **동시에 실행하지 않는다**.
  그 자원의 build·test·압축·다운로드·다른 benchmark·에이전트 계산과도 겹치지 않게 예약한다.
  자기 작업만 제어하고 다른 사용자의 프로세스를 종료하지 않는다. 통제가 불가능한 실행은
  개발용 진단으로 남기고 정식 성능 판정에서 구분한다.
- CPU quota와 affinity를 함께 기록한다. 가능한 경우 SMT sibling·NUMA 배치를 일정하게
  하고 GPU 공유/MPS/MIG·동시 프로세스 여부를 확인한다. affinity나 container만으로
  host CPU·메모리 대역폭·GPU의 독점 사용을 입증했다고 보고하지 않는다.
- 시작 전과 실행 중의 CPU 사용·steal·context switch·cgroup throttle, memory pressure·
  page fault·swap/OOM, GPU 사용·clock·온도·power/throttle를 지원되는 범위에서 기록한다.
  불필요한 heavy profiler는 끄고 monitor 주기와 비용을 양쪽에 동일하게 적용한다.
  관측 불가능한 값은 unknown으로 남기며 핵심 경합 상태가 unknown이면 통제 한계를 명시한다.
- 전력 정책·warm-up·냉각 조건·동시 프로세스 허용 목록과 오염 신호의 허용값은 pilot 후
  확정 측정 전에 고정한다. 통제되지 않은 외부 부하, 설정 이탈, 관측된 외부 자원 압박을
  표시한다. variant 자체의 메모리 증가·throttle·timeout·crash는 회귀 결과로 유지한다.
- 새 process로 시작하는 조건과 model/session/cache가 준비된 steady-state 조건을 분리한다.
  새 process라는 이유만으로 OS page cache나 GPU가 cold라고 부르지 않는다. 전역 cache를
  비우거나 host 설정을 바꾸는 것을 기본 절차로 삼지 않는다.

### 3.3 반복 단위와 순서

첫 실행은 연결·정확성 smoke다. pilot에서는 baseline-baseline과 baseline-variant를 각각
최소 3개 paired block으로 실행해 변동과 필요한 실행 길이를 확인한다. pilot은 후보 선택용이며
확정 결과에 합치지 않는다. 아래 횟수는 이 계획의 기본값이며 GPU/시간 예산 안에서 실행 전에
잠근다. 표본을 줄여 실행했다면 탐색적 결과로 보고하고 같은 확정 근거라고 표시하지 않는다.

| 구분 | 기본 실행 방식 |
|---|---|
| 확정 비교 | 3개 분리된 측정 세션 × 세션당 4개 block = 12개 paired block |
| 한 block | baseline→variant→variant→baseline 또는 반대 순서; 4개의 새 process를 직렬 실행 |
| 순서 배정 | 두 순서를 같은 수로 사용하고 seed로 사전 배정; block 안의 fixture·seed·budget은 같게 유지 |
| 총 실행 수 | 비교 한 개당 48 process; 준비·warm-up 이후 같은 측정 구간 사용 |
| microbenchmark 내부 반복 | timer 해상도보다 충분히 길도록 pilot에서 호출 수를 정하고 확정 때 고정; 내부 호출을 독립 표본 수로 세지 않음 |
| 세션 경계 | 프로세스 정리·자원 preflight·동일 warm-up을 다시 수행; 다른 장비 세션과 합산하지 않음 |
| 재시도 | 객관적 인프라 오염이 있는 block 전체를 새 ID로 재실행; 최대 3개 대체 block, 초과 시 불확정 종료 |

오염 판정은 속도 결과를 보고 유리한 실행만 고르는 방식으로 하지 않는다. 원본·제외 사유·
유효/무효 block 수·재실행 연결을 모두 보존한다. 단지 느리거나 outlier라는 이유로 제거하지
않는다. 통제된 부하 주입은 별도 robustness 실험으로 기록하며 청정 성능 표본에 섞지 않는다.
고정 방문 수 fixture가 정해진 방문 수에 도달하기 전에 timeout/terminal에 도달하면 원인을
기록하고 정상 고정 작업량 성능과 구분한다. variant의 실패를 인프라 오염으로 재분류하지 않는다.

### 3.4 PR #17 observer의 영향

1. 전체 탐색 성능의 주 비교는 양쪽 모두 profile off로 측정한다. 동일한 외부 timer의
   `go` 송신→해당 `bestmove` 수신 구간과 동일 출력 소비 경로를 사용한다. startup·quit·
   보고서 직렬화는 별도 지표다. 내부 source clock과 외부 clock을 섞어 차감하지 않는다.
2. 원인 분석은 양쪽 profile on의 별도 반복 실행으로 수행한다. observer overhead 자체를
   판단할 때는 같은 binary/config의 off/on을 paired 비교하며, 성능 결론에는 위 반복 원칙을
   적용한다. on/off 한 번씩의 결과나 전체 gate 차이로 overhead를 추정하지 않는다.
3. record cap·8MiB 상한·upstream ring을 고려해 profile 실행의 요청 수를 유한하게 정한다.
   snapshot·정렬·JSON 직렬화는 producer 종료 뒤 timed path 밖에서 한다. loss/poison/
   overflow·missing/duplicate stage는 incomplete로 보존하고 해당 phase 분해를 인수하지 않는다.
   observer 불완전성과 실제 엔진 실행 실패는 구별한다.
4. 공유 journal lock이 유의한 비용이라는 근거가 생기면 per-worker buffer 등은 별도
   observer 실험으로 구현한다. cache·polling·batch 변경과 동시에 넣지 않는다.

### 3.5 지표·통계·판정

| 대상 | 비교 지표 |
|---|---|
| A/C 준비 | legal/classify/export/project/encode/key/prepare 시간, history 방문 수, 할당 요청·누적 bytes·live peak |
| 고정 방문 수 탐색 | 같은 완료 방문 수까지의 자기 벽시계, selection→backup 지연, replay 깊이·횟수, 준비 비용 |
| 고정 시간 탐색 | 같은 CPU/GPU/RAM/VRAM·시계 한도의 완료된 유효 traversal, deadline 초과·취소·미소비·착수 |
| 캐시·runtime | hit/miss/eviction/reset, 실제 NN 실행·고유 입력·정상 소비·backup을 각각 집계; ready 대기·queue·batch 분포 |
| 자원 | CPU time·peak RSS와 process VRAM을 구별; 측정 불가 항목은 별도 표시 |

할당 카운터와 heavy/device profiler를 켠 측정은 시간 기준 측정과 분리한다. allocation bytes를
live memory peak로, cache lookup 수를 방문 수나 NN 실행 수로 바꾸지 않는다.

확정 분석은 같은 block의 baseline/variant 비율을 먼저 계산한다. 세션별 결과와 block 비율의
중앙값·분산·95% 신뢰구간을 함께 제시한다. bootstrap을 쓰면 세션별 층을 유지하고 paired
block을 재표집한다. 같은 process의 수백 요청을 독립 표본으로 취급하지 않는다. P50/P95/P99는
요청 지연 분포로 별도 보고하고 표본 부족·세션 간 변동을 명시한다. 분위수 합산이나 유리한
한 실행의 선택을 하지 않는다.

기본 실용 개선 기준은 사전 선택한 1차 비용 지표 **5% 감소**, 전체 고정 작업량 시간의
허용 회귀는 **2%**로 제안한다. 후보의 목적·pilot 변동·예산에 맞춰 확정 실행 전에만 조정한다.
성능 채택에는 paired 신뢰구간까지 해당 기준을 만족하고 정확성·자원 조건을 통과해야 한다.
할당 감소가 1차 지표라면 시간 단축도 입증했다고 보고하지 않는다. 효과가 잡음보다 작거나
신뢰구간이 회귀 허용폭을 넘으면 불확정으로 남긴다. 예산을 늘리거나 결과를 반복 확인하다
유리한 시점에 종료하지 않는다. 작은 세션 수의 결과를 다른 장비의 성능으로 일반화하지 않는다.

탐색 의미를 바꾸는 S 후보의 승격은 별도 holdout·paired 대국 계약을 따른다. 이 계획의
5%/2%는 그 강도 판정 기준을 대체하지 않는다. 기능 CI는 correctness용으로 사용하며 공유
runner의 wall time으로 성능 합격/불합격을 결정하지 않는다.

## 4. 구현 순서와 실험 코드 관리

| 단계 | A가 직접 구현할 산출물 | 다음 단계의 검증 근거 |
|---|---|---|
| 검증 기반 보강 / OPT-00 | 기존 probe를 tensor·전체 fixed-visit trace와 연결하고 유한한 종단 runner를 보강 | 기본/각 옵션/조합 witness; 반복·오염·observer 상태 식별 |
| 기존 내부 실험의 인수 | 이미 구현한 claim·digest의 소비자 회귀·통제 성능 비교 | 구현 상태와 기본 활성화 판정을 분리; 8장 결과는 진단으로 유지 |
| 입력 준비 / OPT-01~03 | 비트보드·반복 정보 → 불변 준비 결과 → hash 호출 축소 | tensor/input key/legal mapping·요청 검증·메모리 상한 |
| 탐색 CPU 비용 / OPT-04~05 | 작은 할당·조회 제거 → bounded 노드 상태 재사용 | f64/tie/오류·selection/leaf/value/backup trace·eviction |
| runtime / OPT-06~07 | exact raw cache → 별도 완료 통지 실험 | computed/hit 계약·정확한 집계·reset·lost wakeup·취소·drain |
| movegen·장치 경계 / OPT-08~10 | 공격 테이블 → pin/check, staging/output pool → 별도 I/O Binding | 규칙 oracle·수명·budget·실제 장치 전송/동기화 |
| 후속 실험 / OPT-11~12 | CUDA Graph와 다중 요청/batch를 독립 구현 | capture/replay 조건; S의 예약·backup·holdout 품질 |

앞 단계의 성능이 불확정이어도 후속 후보의 독립 구현과 CPU/mock 검증은 진행할 수 있다.
미검증 변경을 하나의 성능 variant에 누적하지 않고 각 실험의 기준 소스를 고정한다.
GPU가 필요한 항목도 구현·CPU/mock 검사와 실제 GPU 인수를 분리한다. 실행 예산은
[RunPod 계획](RUNPOD-BENCHMARK-PLAN.md)을 따르며 이 문서로 증액하거나 소비하지 않는다.

**실험 여부와 E/A/S 분류는 별개의 항목이다.** 정확한 결과를 재사용하는 E 가설이어도
상태 저장·요청 준비·완료 전달 등 전체 흐름을 바꾸면 실험 코드로 명시한다. 분류가 미확정이면
`실험 / 의미 보존 미검증`으로 공유하고 구현을 진행한다. 의도한 E 후보에서 선택·평가·backup
순서가 달라지면 단순 최적화 성공으로 처리하지 않고 원인을 규명하거나 S 실험으로 재분류한다.

흐름 변경 실험은 기본 비활성의 명시 옵션/feature로 노출하고 기존 경로를 대조군으로 유지한다.
실험 이름·변경 한 가지·기본값·활성화 방법·영향 consumer·알려진 실패·중단/복귀 방법을 기록한다.
사용하지 않는 경로에서 allocation·lock·worker를 만들지 않는지 확인한다. 실험을 검증하려고
baseline과 variant의 실제 추론을 같은 탐색 안에서 동시에 실행하여 성능을 오염시키지 않는다.
실험 실패를 조용히 기본 경로 성공으로 숨기지 않으며 정상 cache miss의 재계산은 명시 계약대로
처리한다. 새 옵션·공통 필드·revision과 모든 영향 consumer는 A가 함께 구현·검사한다.
총괄 리뷰를 위한 계약 차이·통합 인수 근거를 같은 SHA로 남긴다. 상세 의존성과 완료 조건은
9장을 따른다.

## 5. 구현 PR에 추가할 테스트 항목

기존 검사를 먼저 활용하고 변경이 만드는 실패 경로에 필요한 최소 fixture만 추가한다.
다음은 **실행할 테스트 계획**이며 현재 통과한 항목 목록이 아니다.

| 변경 | 테스트와 비교 대상 | 실패로 볼 조건 |
|---|---|---|
| A claim·history | current/intended 3회·50수, 자동 5회·75수, mate 우선, unknown prefix·irreversible 경계, history/revision/counter 한도 | claim 순서·근거·unknown 또는 기존 실패 단계가 달라짐 |
| A digest·memo | full history/raw EP/counter/origin/completeness, profile·기존 SHA golden, make/unmake·fork·stale/foreign/ABA | 바이트·digest 불일치, 다른 owner/revision view 재사용 |
| A movegen | 기존 perft·독립 python-chess, pin·double check·EP 발견 공격·castling·Q/R/B/N 순서 | 집합/순서·FEN·오류·상태 복원 차이 |
| A/C projection·prepared reuse | 같은 보드/다른 이력, repeated plane, padding/EP predecessor, model/encoding 교체, 불일치 요청 거부 | 실제 tensor·input key·legal mapping 불일치, budget 검증 우회 |
| B 통계·상태 재사용 | 같은 고정 방문 수·동일 seed·가짜 clock/결정적 evaluator에서 selection/leaf/backup/full trace; tie·overflow·eviction | 의도하지 않은 선택/값/소비 차이, 무제한 상태 축적 |
| exact raw cache | cold/hit/eviction/reset, cross-root/cross-game, model/encoding/backend/precision/compute 변경, legal 순서 재검증 | 잘못된 재사용·가짜 physical execution·lookup의 방문 집계·중복 backup |
| 완료 통지·흐름 변경 | 완료 직전/직후 대기, lost wakeup, spurious wakeup, deadline/stop/quit·root/game 교체, 늦은 결과·취소 경쟁·drain | 교착·busy-spin·deadline 위반·stale 소비·물리 완료 전 buffer 재사용 |
| buffer/I/O Binding/Graph | shape/batch 변경·오류·취소·quarantine, 비동기 물리 완료, 장치 주소/수명·반복 Run | 잘못된 출력·미완료 메모리 해제·비용 상한 초과 |
| 다중 요청/batch 또는 S 후보 | virtual reservation·취소 반환·단 한 번의 backup, 예상 selection/완료 순서 차이·자원 상한, 같은 W의 holdout 비교 | 규칙·수명·집계 위반; 의도한 탐색 차이는 기록하고 별도 품질 판정 |
| observer/runner | off/on, cap overflow·poison·missing/duplicate stage, 강제 부하·불완전 journal·실행 실패 분류 | 관측 손실을 complete로 표시, variant 실패를 오염으로 제외 |
| 모든 실험 옵션 | default/off 기존 동작, opt-in 활성화·식별·잘못된 조합 거부, 기존 경로로 복귀 | 기본 실행에 실험 부작용이 유입되거나 결과의 실험 설정을 식별하지 못함 |

E의 비교에서는 digest만 대조하지 않고 ordered moves·전체 이력·classification·tensor 등
해당 witness를 직접 비교한다. process마다 새로 발급되는 ID는 정규화해 내용 비교를 하되
freshness·binding·stale 거부는 별도 검사한다. fixed-visit 동등성은 selection·leaf·value·backup의
의미를 비교한다. 정확한 cache hit로 줄어드는 물리 실행 수, provenance·receipt·시각은
사전에 정의한 기대 차이로 별도 대조한다. 실제 시간 제한에서 완료 방문 수가 증가하는
현상과 고정 방문 수에서 선택 의미가 달라지는 현상을 구별한다. S/A의 의도한 차이는 PR에
허용 범위를 먼저 적으며 규칙·소유권·취소·정확한 집계 조건은 동일하게 지킨다.

## 6. 실험 PR의 표시와 결과 양식

흐름/탐색 변경 PR 제목은 `experiment(scope): ...`로 표시하고 인수 전에는 draft를 유지한다.
본문 첫 문장에 **실험 코드이며 기본 비활성, 의미/성능 인수 상태**를 적는다.
[기존 PR 템플릿](../../.github/PULL_REQUEST_TEMPLATE.md)의 실험 대조·검증 항목에 아래를
추가한다. 아래 항목은 예정된 체크리스트이며 실제 실행 후에만 체크한다.

```markdown
실험 코드 / 기본 비활성 / 의미 보존 미검증 또는 의도한 S·A 변경

- 가설·변경 한 가지·E/A/S 분류와 예상 흐름 차이:
- 기준/변경 source·binary·config·fixture 식별, 고정 W/S와 자원:
- 활성화 옵션·기본값·중단/복귀 방법, 영향 consumer·계약 revision:
- 구현됨 / 부분 구현 / 알려진 실패 / 미실행:

- [ ] 기본 비활성 경로의 기존 동작·비용 유입 검사
- [ ] 변경별 correctness·수명·취소·실패 테스트와 실제 명령/결과
- [ ] E의 witness/고정 방문 trace 또는 S/A의 사전 정의한 차이·품질 검사
- [ ] pilot과 확정 반복 실행 분리; 세션/block/process 수·순서·자원 관측 기록
- [ ] 오염·제외·대체 실행과 제품 실패를 모두 보존
- [ ] profile off 주 비교, on 원인 분석, observer overhead 대조 상태
- [ ] paired 효과·신뢰구간·메모리·deadline·회귀와 미측정 항목 보고
- [ ] GPU/대국이 필요한 항목의 실행 또는 미실행을 명시

결론: 구현 상태 / 정확성 인수 / 성능 인수 / 기본 활성화 여부를 각각 기록
원본: artifact 논리 경로·hash·보존 위치, 이전 자료는 참고 링크로 구분
```

실험 구현을 공유했다는 이유로 기본값을 바꾸거나 제품 성능 개선으로 보고하지 않는다.
정확성만 통과했거나 효과가 불확정인 구현은 실험 상태로 공유할 수 있다. 기본 활성화는
해당 비교·영향 consumer·총괄의 통합 인수 근거를 갖춘 별도 변경으로 검토한다.

## 7. 조사 출처와 완료 범위

- [Stockfish movegen](https://github.com/official-stockfish/Stockfish/blob/master/src/movegen.cpp):
  pin·check 정보를 사용하고 king/pinned/EP 수에 정밀 legality 검사를 적용하는 비교 사례.
- [LC0 search](https://github.com/LeelaChessZero/lc0/blob/master/src/search/classic/search.cc):
  평가 캐시·minibatch·작업 버퍼 재사용 사례. 탐색 정책 차이를 그대로 의미 보존으로 간주하지 않는다.
- [Rust Performance Book: heap allocations](https://github.com/nnethercote/perf-book/blob/master/src/heap-allocations.md):
  실제 할당 profiling, Vec 재사용, Arc clone과 자료 복제의 구별.
- [ORT I/O Binding](https://github.com/microsoft/onnxruntime/blob/935a9bbd79e2194858156c950e6ddc115a0435e8/docs/performance/tune-performance/iobinding.md),
  [CUDA Graph 조건](https://github.com/microsoft/onnxruntime/blob/935a9bbd79e2194858156c950e6ddc115a0435e8/docs/execution-providers/CUDA-ExecutionProvider.md):
  장치 전송·사전 할당·고정 주소·동기화 수명 검토. 제품의 ORT 1.22.0/wrapper 조합에서 지원을
  다시 확인하며 최신 문서의 모든 기능을 현재 지원한다고 가정하지 않는다.

외부 구현은 설계·독립 검증 참고로 사용하고 자체 MIT 코어에 외부 GPL 소스를 복제하지 않는다.
초기 완료 범위는 PR #17의 상태·변경·리뷰·CI 확인과 계획 문서 작성이었다. A의 후속
실험 구현·새 반복 CPU 진단은 아래에 추가한다. 나머지 최적화도 A의 후속 구현 범위로
갱신했으며 controlled 전체 탐색·GPU A/B·실제 강도 인수와 함께 9장에 정리한다.

## 8. A 실험 구현과 실제 검사·CPU 진단 (2026-10-03)

[Draft PR #18](https://github.com/daejunnom/RoveZero/pull/18)에서 첫 두 A 후보를 구현했다.
구현 source는 `5dbd514391467f65ffdfc3af90e98c2731c41164`, 기준은
`b5ba853585cb5a78f81159f733a86cdfe085936b`다. 아래 CPU 결과는 **개발용 진단**이다.
두 옵션의 기본값은 off이며 성능 채택·기본 활성화·통합 인수는 보류한다.

### 8.1 구현과 활성화

| 옵션 | 구현 | 검증·재사용 경계 |
|---|---|---|
| `experimental-claim-preview` | 현재 반복 횟수·증거 완전성을 한 borrowed 순회로 계산하고 intended claim은 checked 전이의 상태·반복 identity를 검사 | 임시 child owner/history를 발급하지 않음; claim 순서·unknown/irreversible·history/revision/counter 실패 순서 보존 |
| `experimental-history-digest` | full history의 borrowed state를 한 103-byte-capacity FEN 버퍼로 직렬화하고 같은 live owner의 현 상태 digest 하나를 `OnceLock`에 보존 | 기존 framing/profile/SHA 바이트 유지; legal/classification을 항상 재검증; 성공 make/unmake에 memo 제거; 새 fork는 빈 memo |

두 옵션은 독립 feature이며 history 옵션은 `contracts`를 활성화한다. 비활성 시 owned
preview와 기존 digest 경로를 사용한다. 공통 전이는 오류 검사 완료 뒤 private generic
소비자에게 전달하여 큰 중간 `Result<CheckedTransition, _>` 반환을 줄였다. 기본 preview는
기존 loop scope에서 해제된다. 공통 계약 revision 0.1, 전체 이력과 owner/revision 권한은
그대로다. 모델 평가 cache·증분 hash codec·history 절단은 구현하지 않았다.
직접 소스 변경은 A crate와 이 연구 기록이며 새 제품 의존성을 추가하지 않았다.

```sh
cargo test -p rz-position --features contracts,experimental-claim-preview --locked
cargo test -p rz-position --features experimental-history-digest --locked
RZ_CHESS_PYTHON="$ORACLE_PYTHON" \
  cargo test --release -p rz-position --all-features --locked -- --include-ignored
```

`ORACLE_PYTHON`은 test-only `python-chess==1.999`, `chess==1.11.2` 환경이다. 두 feature를
제거하면 대조 경로로 돌아간다. B/C/D의 제품 소스와 공통 선언·CI 설정은 수정하지 않았다.

### 8.2 실제 정확성·소비자 검사

Linux x86_64/Rust **1.96.0**, 위 source에서 다음 명령을 실행했다. 이 표는 서로 다른
feature 설정의 실행 결과이며 합산 고유 테스트 수가 아니다.

| 명령 | passed / failed / ignored |
|---|---|
| `cargo test -p rz-position --locked` | 29 / 0 / 2 |
| `cargo test -p rz-position --features contracts --locked` | 42 / 0 / 2 |
| `cargo test -p rz-position --features contracts,experimental-claim-preview --locked` | 44 / 0 / 2 |
| `cargo test -p rz-position --features experimental-history-digest --locked` | 45 / 0 / 2 |
| 위 release/all-features/include-ignored 명령 | **49 / 0 / 0**, 확장 perft·독립 oracle·doctest 포함 |
| `cargo test --workspace --all-targets --all-features --locked` | **680 / 0 / 16** |
| `cargo fmt --all --check` | 성공 |
| `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | 성공 |

추가한 검사는 borrowed claim과 실제 owned child의 증거·오류 비교, 원본/live-view 보존,
FEN 기존 바이트·버퍼 capacity 보존, make/unmake/실패/새 fork의 digest와 stale 권한,
4개 thread의 최초 export 동시 초기화를 다룬다. 기존 SHA golden·수 순서·full-history·
claim/자동 종료·mate 우선·overflow 검사를 함께 유지했다. Workspace에는 실제 Rules→
projection 및 UCI/mock search 연결 검사가 포함되지만 고정 방문 전체 trace의 feature 간
동등성 검사는 별도 미실행이다. Workspace의 16 ignored를 통과로 세지 않았다.

[동일 구현 source CI 37130856771](https://github.com/daejunnom/RoveZero/actions/runs/37130856771)의
Ubuntu·Windows job과 fmt/workspace/native-cli/독립 oracle/F fixture/Clippy 필수 step 성공을
확인했다. 이는 GPU·실제 weights 추론·대국 성과의 증거가 아니다.

[`performance_probe`](../../crates/rz-position/examples/performance_probe.rs)는 `contracts`
필수의 유한한 CPU 예제다. 제품 실행에는 연결되지 않는다. 기준 checkout에도 같은 예제와
fixture만 추가해 빌드했다. 원본·현재 default·claim만·history만·두 옵션 조합의 `witness`
출력 전체가 byte-for-byte 일치했다. **14개 root, 253개 child, 787개 상태 기록**이며
재-export/make/복원 기록을 포함한다. full newest-first FEN, origin/completeness/revision,
ordered moves·order digest, 구체·공통 classification, semantic digest, terminal WDL을
비교하고 stale view 거부를 확인한다. owner 발급값은 출력에서 정규화하며 freshness는
별도 검사한다. 동일 출력 SHA-256은 다음과 같다.

`a3fe4a6c4726bcdc342141ba2749e6a09ab8a0f2a27c5e293718a1f6ea6011ba`

### 8.3 반복 비교의 설정과 한계

이력 1/17/65/129/257의 진행 상태를 고정했다. 저장된
[`256-ply trace`](../../crates/rz-position/tests/fixtures/performance_trace.txt)는 독립 참조의
합법 fixture(seed 0)이며 terminal 뒤에 수를 이어 붙이지 않는다. `classify`, 새
owner+export, 같은 상태의 warm export, attested fork+export를 구분했다. 첫 export와
warm-up은 warm-export의 timed region 밖이다. 새 owner·fork는 항상 새 memo이며 cold
표시는 digest memo의 miss를 뜻한다. OS cache·process startup cold를 뜻하지 않는다.

공통 설정은 Rust 1.96.0 release/locked, System allocator, Intel Xeon Platinum 8573C,
Linux 6.18.44/glibc 2.41, CPU 허용 0–4·quota 4 core·RAM 16GiB다. child는 CPU 0,
수집기는 CPU 4에 고정했다. 빌드·검사·압축을 모두 끝낸 뒤 비교 process를 직렬 실행했다.
GPU와 ONNX·모델 세션·search journal은 생성하지 않았다. PR #17 observer off/on 효과를
측정한 결과로 사용하지 않는다. host 독점·실제 clock/SMT/NUMA 경합은 입증하지 못했다.

첫 source `29f4848`에서 claim-only의 긴 이력 export 시간 회귀를 관측했다. 그 자료를
`revision-29f4848/`에 보존한 뒤 중간 반환과 default preview scope를 수정하고 별도 회차를
실행했다. **두 회차를 합산하거나 앞 회차를 제외한 확정 성능으로 보고하지 않는다.**
첫 회차의 긴 이력 claim cold/warm/fork 비율은 각각 1.245/1.270/1.324였다. 최신 회차의
변동·구현 차이만으로 그 회귀가 해결됐다고 판정하지 않는다. 첫 연결 실행은 수집기의
Python API 오타로 중단됐으며 child 출력·오류·수정 전 수집기를 보존하고 pilot 전에
고쳤다. 제품 실패나 느린 표본의 제외로 처리하지 않았다.

각 회차에서 baseline-baseline 및 세 비교(original→default, default→claim만,
default→history만)의 pilot을 각각 3 block×4 fresh process로 실행했다. 본 비교는
비교별 **3개 측정 창×4 block×4 fresh process = 48 process**, ABBA/BAAB를 창마다
2개씩 seed 20261003으로 배정했다. 회차별 pilot 48, 본 비교 144, 총 192 process다.
측정 창은 같은 VM의 연속 구간이며 세션 경계에서 preflight와 같은 warm-up을 재개했다.
내부 반복은 표본 수로 세지 않는다. pilot의 timer 구간만으로 호출 수를 결정하고 본
비교 전에 고정했다. 앞 회차는 1,226회, 최신 회차는 1,483회이며 warm-up은 지표당 20회다.
호출 수와 회차를 섞은 직접 시간차로 수정 효과를 판정하지 않는다.

각 process 전후에 cgroup CPU·memory event·압력/steal 관측값을 보존했고 wait4로
child CPU time/RSS/fault/context-switch를 기록했다. CPU PSI·host 독점은 unknown이다.
최신 본 비교에서 관측한 throttle/OOM kill/major page fault 증분은 0, child 비자발적
context switch 중앙값 4·범위 0–22였다. 이 관측으로 host 경합이 없었다고 판정하지 않는다.
pilot의 같은 binary 긴 이력 cold 비율도 0.731–1.077로 변동했다. 전체 process를 보존했고
느린 block 제외·대체·추가 횟수 연장은 하지 않았다. RSS는 fork/pre-exec·준비·warm-up을
포함하는 process high-water이므로 상태 cache의 live memory peak로 해석하지 않는다.
이 한계 때문에 **정식 성능 판정과 2% 전체 탐색 회귀 기준을 인수하지 않는다.**

### 8.4 최신 회차의 CPU 시간 진단

각 block의 두 variant 시간 평균 / 두 baseline 시간 평균을 계산했다. 아래는 12개
paired block 비율의 중앙값이며 **1보다 작으면 해당 API 지표가 짧다**. 기본 경로의
공통 코드 정리도 원본과 별도로 비교했다. 일부 지표의 비용 증가는 그대로 유지한다.

| 비교 | 이력 | 합법 수 | classify | 새 owner+export | warm export | fork+export |
|---|---:|---:|---:|---:|---:|---:|
| 원본→default | 1 | 20 | 0.983 | 1.039 | 1.029 | 1.067 |
| 원본→default | 17 | 28 | 1.073 | 1.056 | 1.027 | 1.077 |
| 원본→default | 65 | 41 | 0.931 | 0.860 | 0.934 | 0.921 |
| 원본→default | 129 | 24 | 0.925 | 0.903 | 0.936 | 0.942 |
| 원본→default | 257 | 37 | 0.942 | 0.997 | 0.898 | 1.076 |
| default→claim만 | 1 | 20 | 0.340 | 0.589 | 0.484 | 0.584 |
| default→claim만 | 17 | 28 | 0.432 | 0.692 | 0.664 | 0.683 |
| default→claim만 | 65 | 41 | 0.420 | 0.843 | 0.827 | 0.954 |
| default→claim만 | 129 | 24 | 0.380 | 1.030 | 0.976 | 1.021 |
| default→claim만 | 257 | 37 | 0.533 | 0.942 | 0.974 | 0.806 |
| default→history만 | 1 | 20 | 1.058 | 1.101 | 0.994 | 0.958 |
| default→history만 | 17 | 28 | 1.034 | 0.818 | 0.541 | 0.792 |
| default→history만 | 65 | 41 | 1.063 | 0.633 | 0.320 | 0.591 |
| default→history만 | 129 | 24 | 1.034 | 0.672 | 0.136 | 0.580 |
| default→history만 | 257 | 37 | 1.138 | 0.504 | 0.098 | 0.396 |

다음은 사전 선택한 1차 지표와 warm/fork 보조 지표다. 시간은 각 설정의 process 내부
ns/call 값의 중앙값을 µs로 표시했다. 비율은 위 paired 통계라 시간 중앙값의 단순 나눗셈과
다를 수 있다. 95% 구간은 세 측정 창의 층을 유지한 paired-block bootstrap 5,000회,
seed 20261003, 비율 중앙값의 percentile 구간이다. 다중 지표 보정 없는 기술 통계이며
반복 호출·process를 독립 block으로 세지 않았다. 모든 20지표×3비교의 block 원본,
비율 분산·창별 중앙값·구간은 `summary.json`에 보존한다.

| 비교·이력 257 지표 | baseline µs | variant µs | paired 비율 | 95% 구간 |
|---|---:|---:|---:|---|
| claim / classify | 12.12 | 6.53 | 0.533 | [0.493, 0.565] |
| claim / cold-owner-export | 138.41 | 143.27 | 0.942 | [0.881, 1.257] |
| history / cold-owner-export | 138.39 | 75.70 | 0.504 | [0.488, 0.562] |
| history / warm-export | 157.30 | 15.00 | 0.098 | [0.093, 0.112] |
| history / fork-export | 168.14 | 69.60 | 0.396 | [0.366, 0.410] |

claim-only의 긴 이력 첫 export 구간은 회귀까지 포함한다. history-only도 이력 1의
첫 export 비율 1.101, 구간 [1.000, 1.184]를 보였다. 기본 경로 대조의 여러 구간도
허용 회귀폭을 넘는 변동을 포함한다. 시간 측정의 개선 징후·회귀 가능성을 모두 진단으로
남기며 기본 활성화를 하지 않는다. 두 옵션 조합의 시간 효과와 실제 fixed-visit 전체
탐색의 5%/2% 기준은 미측정이다.

### 8.5 시간 측정과 분리한 할당 진단

제품 밖의 별도 probe에서 System으로 그대로 전달하는 전역 allocation counter를
사용했다. 각 설정 3 fresh process, timed API 구간 대신 100회 호출의 alloc/alloc_zeroed/
realloc **요청 수·요청 byte 합계**만 비교했다. 이 계측의 시간값을 위 시간 자료에 합치지
않았다. 5개 fixture×4 API×4 설정의 80개 조합에서 세 실행의 count가 각각 일치했다.
카운터의 unsafe allocator forwarding은 외부 진단에만 있으며 제품 A 코드에는 없다.

아래는 이력 257, 합법 수 37에서 호출당 `요청 수 / 요청 bytes`다. 요청 bytes는 live
peak나 복사 bytes와 다르다. 이력 1의 claim classify도 44/8,207→4/207이었다.

| API | default | claim만 | history만 | 두 옵션 |
|---|---:|---:|---:|---:|
| classify | 76 / 15,002 | 2 / 202 | 76 / 15,002 | 2 / 202 |
| cold-owner-export | 1,896 / 53,914 | 1,822 / 39,114 | 89 / 16,489 | 15 / 1,689 |
| warm-export | 1,895 / 53,898 | 1,821 / 39,098 | 87 / 16,370 | 13 / 1,570 |
| fork-export | 1,867 / 46,655 | 1,829 / 39,055 | 53 / 9,097 | 15 / 1,497 |

할당 감소는 이 유한한 fixture의 직접 관측이다. 전체 탐색 속도·GPU 활용·강도 향상이나
cache memory peak를 함께 입증한 것으로 보고하지 않는다.

### 8.6 원본 보존과 다음 인수

원시 자료는 저장소 밖 `${ARTIFACT_ROOT}/position-perf-impl/`에 보존했다. 최신 자료의
`measurement-manifest.json`, `processes.jsonl`, `timing/`, `summary.json`, `checks.json`,
`check-*.log`, `witness-*.txt`, `probes.json`, `allocation-counts.json`, 별도 allocator source와
수집/분석 script를 유지한다. 첫 회차는 `revision-29f4848/`, 연결 수집기 실패는 그 안의
`collector-setup-attempt/`다. 동일 질문의 기록을 이 문서에 묶고 원시 로그를 source에
넣지 않았다. 두 회차·probe·fixture·구현 diff를 포함하는 회수용 archive
`position-performance-evidence-20261003.tar.gz`와 SHA-256 목록을 작업 output root에
보존한다. 이 세션의 output root를 삭제하지 않으며 장기 외부 보관은 별도 회수한다.

최신 feature별 binary hash는 다음과 같다. fixture hash는
`004cc5a461266258c38aa32cbae7aa34ea5174eb5899c507d683d91911b7a4c2`,
probe source hash는 `19ca8fce694a40e838944bf7c28c208217116454d629d514f87b3d894ad35841`다.

| binary | SHA-256 |
|---|---|
| original | `136cd9a0621f62a96c113343ed8c089c06f51e28f2f86a58f0f28e2df655540d` |
| default | `a544ded1cc716fe525f4a9854d9ca7181f99466cbd940b99e9bdd6bc9b55de38` |
| claim | `15ae964d4813edc9c426da3c002703db288d25dc0a47ea17413568e41f61fe21` |
| history | `cd05e345c3575c1651b423b320e4f353c3c2ba30fb7c9ce15c5ce3ae28e76803` |
| both | `b326350ad5f0e8b90faf504b87586c71cf7673946349ce797cc8c82af08eb2a8` |

남은 인수는 통제 장비의 전체 fixed-visit/selection/leaf/value/backup trace, 실제 C
tensor/input key·B/D 시간/취소/물리 수명과 결합한 회귀 비교, observer off/on 비용,
실제 모델·GPU와 동일 자원의 대국 강도다. 8장에 기록한 source `5dbd514`에는 B/C/D 후보
및 movegen/projection 개선이 포함되지 않았다. 최신 배정으로 이 항목도 A가 구현할
9장의 후속 범위에 포함한다. 기본 활성화 전에는 allocation 감소와 짧은 API 개선만으로
이를 대체하지 않고 A가 소비자 검증을 연결하여 총괄 리뷰용 integration 근거를 남긴다.

## 9. 전체 최적화의 미반영 항목과 실행 계획

### 9.1 구현됨·미반영·인수 대기의 구분

계획 갱신 전인 2026-10-03 UTC에 PR #18 `15b9d50`과 PR #17 `81059e4`가 모두 open/draft이며 develop은
`b5ba853`임을 재확인했다. 두 head의 Ubuntu·Windows CI 필수 step은 성공했고 미해결
inline review thread는 없다. 이후 구현의 manifest는 당시 최신 full SHA를 다시 고정한다.

| 항목 | 현재 반영 상태 | 남은 일 |
|---|---|---|
| #16의 legal 중복 생성·부모 재-export·discarded undo·profile hash | develop에 병합됨 | 기준선에 포함; 같은 개선을 다시 구현하지 않음 |
| claim 반복 증거·임시 전이 축소 | #18의 `experimental-claim-preview`, 기본 off; 10장의 CPU 종단 witness 일치 | 통제 성능·실제 모델 비교·통합 및 활성화 판단 |
| FEN 버퍼·현 상태 digest 재사용 | #18의 `experimental-history-digest`, 기본 off | 같은 인수; 짧은 이력 회귀 가능성도 재확인 |
| 유한한 public probe·상태 witness·반복 CPU/할당 진단 | 구현·실행됨, 8장에 증거 보존 | 기존 숫자를 새 비교 표본으로 사용하지 않음 |
| OPT-00 실제 tensor·고정 방문 수 종단 witness·직렬 UCI runner | CPU/mock 구현·대조·실행됨, 10장 | 실제 UCI 완료 방문 수 관측·통제 세션·observer off/on·native 실행 |
| D02 source observer | #17에 구현됐지만 현재 기준선에는 미병합 | 사용할 통합본의 선언·consumer 대조, off/on overhead와 관측 손실 검사 |
| 아래 OPT-01~12 | 내부 구현·소비자 연결·CPU 검사 완료; 11장 | 전부 A가 구현·소비자 연결·검증; GPU/정식 성능·품질 인수는 별도 |

OPT 번호는 이 계획의 작업 단위이며 기존 TASK/CARD 배정을 바꾸는 번호가 아니다.
다음 표는 최초 **구현 계획**과 의존성이다. OPT-01~12는 실제 소비 경로까지 구현했으며
현재 구현·검사·인수 상태는 11장에 기록한다. 지원 옵션의 선언이나 backend의 batch
capability만으로 실제 탐색 연결·GPU·품질 검증을 완료했다고 세지 않는다.

### 9.2 우선순위와 의존성

| 단위 | 미반영 산출물·주 변경 파일 | 선행 및 분리 조건 | 완료 근거 |
|---|---|---|---|
| OPT-00 / CPU 도구 구현됨 | 실제 tensor·fixed-visit witness, 유한 종단 비교 runner; uci 예제·runtime Python 검사·CPU CI | 가짜 clock correctness와 실제 시간 성능 분리; 10장 | 5설정×2 history fill 출력 일치, 584회 CPU 진단; 정식 인수는 남음 |
| OPT-01 | 읽기 전용 비트보드·borrowed 이력 반복 정보; position `position.rs/types.rs`, eval `rules_projection.rs` | 비트보드 읽기와 반복 계산을 각각 비교 | 전체 입력 바이트·repeated plane·padding·합법 수 순서 일치 |
| OPT-02 | projection/encoded/input key/legal indices의 불변 준비 결과; eval `rules_projection.rs/contracts.rs`, search/uci 연결 | OPT-01 이후; 공유 객체와 새 요청의 검증을 분리 | 중복 encode 제거, 잘못된 요청·profile·budget 거부 유지 |
| OPT-03 | input key의 작은 hash update 축소; eval `contracts.rs`, encoding 접점 | OPT-02와 별도 variant | 기존 little-endian framing·golden·key 일치 |
| OPT-04 | PUCT·통계 작업 버퍼·best move 조회; search `policy.rs/tree.rs/contracts.rs`, uci `engine.rs` | OPT-00; 조회/argmax/버퍼를 각각 작은 비교로 분리 | tie·오류 우선순위·전체 고정 방문 trace 일치 |
| OPT-05 | bounded 노드 상태 재사용; search `tree.rs/contracts.rs`, Rules adapter | OPT-00; OPT-04와 분리 | replay 대조·eviction·세대/수명·메모리 상한 |
| OPT-06 | 같은 game의 exact raw-eval cache; eval/native bridge·runtime/scheduler·search 소비자 | OPT-02·00; notifier 없이 먼저 비교 | 실제 hit/miss·fresh 요청·RawEvalHit/receipt·한 번의 backup |
| OPT-07 | 완료·취소 알림과 deadline 대기; eval `worker.rs/native_runtime_bridge.rs`, runtime·uci | OPT-06과 별도 변경/대조; cache off에서도 검사 | lost wakeup·stop/quit·stale·physical drain·대기 지연 |
| OPT-08 | 공격 테이블 → pin/check movegen; position `movegen.rs` | 두 변경 분리; claim/digest 옵션도 각각 대조 | 기존 perft·독립 oracle·move 순서·EP/castling 오류 보존 |
| OPT-09 | 입력 staging·raw output buffer 재사용; eval `onnx.rs/worker.rs`, runtime ledger | 모델/shape/physical completion 경계 유지 | 할당·peak memory·출력·quarantine 검사 |
| OPT-10 | ORT I/O Binding; eval `onnx.rs`, native loader/bridge의 capability·수명 접점 | OPT-09; 실제 ORT 1.22.0/rc.10 지원 확인 | 장치 전송/Run/동기화 분리 계측·실제 GPU 대조 |
| OPT-11 | CUDA Graph capture/replay | OPT-10; Graph 단독 variant | 고정 shape/주소·오류·수명·실제 GPU 반복 실행 |
| OPT-12 | 탐색 다중 진행 요청·batch·virtual reservation; search·runtime·eval·uci | OPT-06/07/09 이후; Graph 채택은 선행 조건이 아님 | 정확한 예약/backup·유한 대기·S trace·동일 자원 holdout |

이 순서로 OPT-01~12를 구현하고 독립 옵션과 조합의 실행 비교를 직렬화했다.
기존 두 feature의 정식 인수와 새 GPU/품질 검증은 별도 상태로 유지한다. 단일 변경의
진단과 조합의 진단을 각각 보고하며 독립 옵션의 효과를 조합의 효과로 대신하지 않는다.

### 9.3 OPT-00과 기존 두 실험의 남은 검증

- 기존 합법 256-ply fixture와 짧은/긴 이력·claim·unknown-prefix 입력군을 사용한다.
  원본/default/claim/history/두 옵션에서 ordered moves·full history·classification·digest뿐
  아니라 C의 실제 tensor/input key/legal indices를 비교하는 유한 witness를 추가한다.
  이 CPU/mock 대조는 10장에서 완료했으며 이후 각 variant의 회귀 검사에 재사용한다.
- 같은 seed·고정 방문 수·결정적 evaluator와 가짜 clock으로 selection/leaf/value/backup 및
  최종 root 통계를 비교한다. stop/deadline/stale/overflow/resource 오류는 별도 시나리오로
  검사한다. ID 발급 차이만 정규화하고 권한·중복 소비 오류는 정규화로 숨기지 않는다.
  10장의 witness는 cancel/expiry/stale/node limit/input/output 오류를 검사했고 overflow는
  기존 독립 A 회귀 검사에서 검증했다. 실제 모델과 통제 장비의 소비 trace는 아직 남아 있다.
- 실제 clock의 `go`→`bestmove` 비용·완료 방문 수·deadline·CPU·peak RSS를 수집할 runner를
  기존 실험 manifest와 연결한다. CPU/mock, 실제 CPU 모델, 목표 GPU 모델은 서로 다른
  cohort로 남긴다. load/warm-up/quit와 observer 직렬화는 탐색 구간 밖에서 별도 기록한다.
  구현된 runner는 position 준비까지 포함한 지표와 go 구간을 분리한다. 현재 mock UCI의
  실제 완료 방문 수 출력은 없어 null로 보존하며, 방문 처리량 검증은 후속으로 남긴다.
- pilot과 사전 고정한 3세션×4 paired block, ABBA/BAAB의 48 fresh process 비교를 적용한다.
  baseline-baseline과 profile off/on 대조도 유지한다. 경합을 통제·관측하지 못하면 진단으로
  표시한다. 첫 두 옵션을 각각 비교하고 조합 효과는 별도 비교로 확인한다.

기존 feature의 CPU/mock 고정 방문 수 전체 동일성 검사는 10장에서 완료했다.
**실제 모델 대조**, **통제 성능 비교**, **기본 활성화 판단**은 남아 있다.
8장의 정확성 검사 성공을 취소하지 않으며 그 근거를 전체 탐색 인수로 확대하지
않는다. PR #17 observer는 실제 선택한 통합 SHA에서만 사용한다. lock 비용이 문제로
측정될 때 per-worker journal 같은 변경을 별도 후보로 만들고 cache/notifier와 묶지 않는다.

### 9.4 OPT-01~03: 입력 준비의 반복 계산 제거

OPT-01의 첫 변경은 A가 이미 유지하는 12개 bitboard를 읽기 전용으로 제공해 C의
64-square 재구성을 없애는 것이다. 두 번째 변경은 최근 최대 8개 frame의 반복 여부를
borrowed 이력 한 순회에서 판별한다. frame마다 **그 frame보다 오래된 known prefix**만
비교해야 한다. 최대 8개 target에 필요한 유한한 저장 공간부터 사용하고 전체 이력 크기의
무제한 map을 만들지 않는다. 순회·identity 복제·할당을 각각 계측한다.

C의 `repeated`는 더 오래된 동일 repetition identity가 한 번이라도 있는지 판별한다.
claim의 3/5회 기준·irreversible 경계·증거 완전성과 그대로 호환되는 값이 아니다.
legal EP를 포함한 동일 identity를 사용하고 origin/unknown prefix·raw EP·history fill을
보존한다. 이력을 최근 8개로 잘라내지 않는다. 같은 보드와 최근 frame을 가지지만 과거
이력만 다른 fixture, EP predecessor, 승격·make/unmake·fork를 기존 경로와 직접 대조한다.

OPT-02는 trusted encoder가 실제 불변 Rules 상태에서 만든 projection·dense tensor·key·
ordered policy indices를 하나의 준비 결과로 묶는다. 먼저 내부 객체를 구현한 뒤 input-key
콜백과 runtime prepare 소비자를 연결한다. 객체의 생성 경로·필드를 제한하고 수명/용량을
명시해 key만 같은 임의 tensor를 받아들이지 않는다. 요청마다 model/encoding/backend/
precision/compute/history profile·현재 position/legal binding·host byte budget을 검사한다.
같은 tensor를 다시 encode하는 비용은 제거하되 실제 입력과 요청의 일치 검증은 유지한다.
공유 준비 결과의 bytes도 상한에 포함하고 root/game/model 교체와 stale 요청을 검사한다.

OPT-03은 key의 byte codec을 유지하면서 작은 `hash.update`를 유한한 작업 버퍼로 묶는
독립 변경이다. f32 little-endian 순서·frame/encoding framing·기존 golden을 유지하고
digest 형식이나 precision은 바꾸지 않는다. projection/encode/key/prepare의 호출 수·
allocation·시간, 전체 request 준비 시간을 함께 비교한다.

### 9.5 OPT-04~05: 탐색 할당과 상태 재생성 제거

OPT-04는 PUCT의 점수 Vec 없이 argmax를 계산한다. parent visit 합의 overflow 검사를
먼저 수행하고 모든 edge의 유효성 검사를 유지한다. f64 식의 연산 순서·strict `>` tie와
뒤쪽 edge 오류를 보존한다. 범용 SelectionPolicy가 받는 통계는 재사용 작업 버퍼로
제공하고 유효 수명 동안 덮어쓰지 않는다. 이 변경과 UCI의 가벼운 best-move 조회 API를
각각 비교한다. 조회는 root 준비 전/방문 0의 None·기존 tie를 유지하며 전체 outcome과
policy 문자열은 최종 결과나 실제 진단 요청에서만 만든다.

OPT-05는 노드에 연결된 검증된 불변 Rules 상태를 root 수명 안에서 재사용한다. node ID와
generation을 묶고 entry/node/byte 상한·eviction·root/game/model 교체를 구현한다. miss는
기존 경로로 재생성하며 hit도 현재 authority/terminal/legal guard를 거친다. board/FEN만
같은 다른 이력으로 교체하지 않는다. 공유 history의 실제 보존 비용과 peak RSS를 기록하고
runtime이 보유한 snapshot/physical lease가 끝나기 전에 강제로 해제하지 않는다.

무캐시/상한 0/작은 상한/eviction/깊은 경로를 같은 고정 방문 trace로 비교한다. replay·
export 횟수, selection 준비 시간, allocation·메모리가 1차 진단 지표다. lookup/hit를 방문,
selection 또는 평가 횟수로 더하지 않는다.

### 9.6 OPT-06~07: exact raw cache와 완료 대기를 별도 구현

OPT-06은 현재 `RawOutput`의 **전체 policy logits와 WDL**을 유한한 entry/byte 상한으로
저장한다. legal-filtered prior나 subtree를 cache하지 않는다. 같은 game의 연속 root에서
사용하며 `ucinewgame`과 model/encoding/backend/precision/compute profile 교체에 초기화한다.
키는 실제 tensor bytes/codec/history fill과 모든 계산 조건을 식별하고 hash collision으로
다른 입력을 재사용하지 않게 equality 근거를 보관한다. 저장·대조 비용도 계측한다.

현재 공통 선언에 `RawEvalHit { source_execution }`은 있지만 fresh runtime은 computed-only다.
따라서 lookup 구현과 함께 C의 새 legal view 변환, D admission/finalization/receipt, B의
최종 guarded backup을 모두 연결한다. hit는 새 request/selection/context를 사용하고
원 계산의 provenance를 기록하며 **새 physical execution은 None**으로 둔다. 실제
미실행 작업의 execution/worker/GPU 시각을 만들지 않는다. 실제 물리 실행 수·hit 수와
논리 selection/accepted visit/backup 수를 분리한다.

새 요청의 legal order·모델·취소·deadline·generation을 lookup 및 최종 소비 경계에서
검사하고 host budget 검사를 우회하지 않는다. 현재 game/model scope에서 검증된 성공
출력만 적재하며 실패·quarantine·stale 쓰기·취소 경쟁을 검사한다. cold/miss/cross-root hit/
eviction/reset, 입력은 같지만 조건이나 이력이 다른 경우, 해시 충돌 주입을 대조한다.
cache off/hit에서도 정상 selection당 backup은 한 번이다. 기존 72/258은 후보 근거이며
이 실험의 hit rate나 절약량으로 사용하지 않는다.

OPT-07은 UCI의 Waiting 1ms sleep을 완료·취소·scope 변경 알림과 deadline을 함께 기다리는
방식으로 바꾼다. `pump`와 결과 소비는 nonblocking으로 유지하고 orchestration owner에서만
유한하게 기다린다. 등록 뒤 predicate/sequence를 다시 확인해 완료 직전·직후 lost wakeup을
막고 spurious wakeup·여러 사건의 병합을 처리한다. 결과 receipt의 소유권은 기존 owner에
남기며 알림 수신자가 결과를 먼저 빼앗지 않는다. soft/admission/hard deadline의 용도를
보존하고 stop/quit/root 교체도 대기를 깨운다.

논리 취소는 physical 완료가 아니므로 기존 drain/quarantine/lease를 유지한다. cache
off/on에서 race fixture를 각각 검사하되 주 성능 비교는 notifier 변경만 사용한다.
worker 완료→ready→backup 지연, wakeup/poll 수·CPU 사용·deadline을 비교한다. 같은 고정
방문 의미가 달라지면 원인을 고치거나 S 실험으로 명시한다.

### 9.7 OPT-08: 합법 수 생성의 정밀 검사 범위 축소

공격 기하 테이블과 pin/check 기반 생성은 별도 opt-in 변경으로 진행한다. 먼저 pawn/
knight/king·ray 기하 계산을 줄이고 그다음 checkers·pinned·evasion mask로 후보를
제한한다. king 이동·EP 발견 공격·castling·pinned move는 필요한 정밀 공격 검사를
유지한다. double check·rook 권리 상실·legal EP repetition identity도 검사한다.

기존 generator를 대조군으로 유지하고 ordered moves, Q/R/B/N 승격 순서, child FEN/
classification·claim·digest·make/unmake·overflow를 비교한다. 기존 확장 perft와 독립
python-chess oracle, 유한한 합법 게임 trace를 사용한다. pseudo/정밀 검사/clone 수·시간과
전체 탐색을 함께 측정한다. 외부 GPL 구현은 참고만 하고 자체 구현한다.

### 9.8 OPT-09~12: ONNX 밖의 buffer·장치·탐색 경계

OPT-09는 요청/shape별 input staging과 raw output 작업 버퍼를 bounded pool로 재사용한다.
현재 `active_input`과 CUDA 오류의 session/input quarantine를 유지한다. 물리 완료가
입증된 slot만 반환하고 host/device/pinned 예약과 pool 보존 메모리를 집계한다. 출력
소비자가 보유 중인 값을 덮어쓰지 않는다. shape/batch 변경·allocation 실패·출력 오류·
취소·shutdown·quarantine를 검사하고 복제 bytes·allocation·peak RSS/VRAM을 비교한다.

OPT-10은 실제 ORT/wrapper 지원을 확인한 I/O Binding 경로를 별도 구현한다. 입력/출력
장치 주소·shape·session·stream·동기화와 실행 수명을 고정하고 지원하지 않는 capability는
명시 오류로 남긴다. CPU/mock으로 경계 코드를 검사하고 목표 GPU에서 전송·kernel·동기화를
분리 측정한다. Run host interval만으로 장치 절약량을 주장하지 않는다. 입력 바이트와
policy/WDL의 사전 고정 수치 기준, 소비 trace·deadline·메모리도 비교한다.

OPT-11은 고정 shape/batch·buffer 주소·session 조건의 Graph capture/replay를 독립
실험한다. warm-up/capture 비용도 전체 자기 시간·메모리에 포함한다. 조건 변경 시
invalidation, capture/replay 실패와 quarantine, 비동기 fence 이전의 재사용을 검사한다.
실제 GPU 반복 실행과 해당 버전의 지원 증거가 없으면 인수 미완료로 남긴다.

OPT-12는 현재 B의 단일 pending selection을 다중 ticket/virtual reservation으로 확장하는
**S 실험**이다. 기존 runtime/backend의 batch 지원을 재사용하되 실제 search 연동을
구현한다. 진행 요청·queue·batch size·max wait·메모리·deadline을 제한하고 각 완료/취소/
오류/캐시 hit에서 reservation을 반환하며 selection당 backup을 한 번만 수행한다.
selection/완료 순서의 예상 차이를 먼저 PR에 명시한다. 고정 방문 수에서 S0와 같다고
가정하지 않으며 같은 W의 holdout·동일 CPU/GPU/시간/메모리 paired 대국으로 별도 판단한다.
Graph·batch 확대·precision 변경을 한 variant에 함께 넣지 않는다.

### 9.9 공유·인수·범위의 완료 조건

현재 문서와 첫 두 실험은 Draft #18에서 유지한다. 후속도 작은 의미 단위로 commit/push하고
같은 목표의 branch/PR를 재사용한다. 독립 비교나 충돌 격리가 필요할 때만 별도 branch/PR로
나눈다. 각 흐름 변경 PR에 실험/default off·영향 consumer·계약 차이·예상 의미 차이와
5장의 해당 테스트를 적고 실제 실행 후에만 완료 표시한다.

각 단위의 완료 기록은 **내부 구현 → 소비자/계약 연결 → 정확성 → 반복 성능 → 장치/품질
인수 → 기본 활성화 판단**을 구분한다. 성능 효과가 불확정이면 구현을 공유하되 기본값을
바꾸지 않는다. 사용하지 않는 실험의 allocation/lock/worker 유입도 검사한다. 허용치는
3장의 사전 기준을 사용하며 음성 결과와 실패를 보존한다.

9장의 계획 갱신은 제품 코드를 추가하지 않았고 10장은 별도 검증 도구의 구현을 기록한다.
학습·가중치 교체·근사 cache·history 절단·
증분 digest codec은 포함하지 않는다. GPU 실험의 코드와 검사 진입점은 구현 범위에
포함하지만 실제 실행은 지정 장비와 기존 예산 안에서 한다. GPU 부재나 새 자원 미확정은
CPU/mock 구현을 막지 않으며 미실행 GPU·강도 결과를 통과로 표시하지 않는다.

## 10. OPT-00 구현·검증과 반복 CPU 진단

### 10.1 실제 실행 소스와 검증 도구

2026-10-03 UTC에 PR #18과 #17의 open/draft·head/base·inline review·필수 CI를
재확인한 뒤 OPT-00을 구현했다. 실행 기준은 develop
`b5ba853585cb5a78f81159f733a86cdfe085936b`, 구현·runner source는
`05c6e5febc33d95a13d5e44128c645bd8fa79d55`다. PR #17 `81059e4`는 미병합
참고 자료로 유지하며 이번 source에 observer를 추가하지 않았다.

[`optimization_witness`](../../crates/rz-uci/examples/optimization_witness.rs)는
실제 A Rules adapter, C `ClassicalProjection`/`MaiaBinding`, D `ContractEvaluator`와
scripted backend, B `ContractSearch`/PUCT를 연결한다. 수동 clock과 명시적 합성
logits/WDL을 사용하여 정확성을 검증한다. 실제 NN 실행이나 시간 성능 표본은 아니다.
제품 library·공통 계약 revision 0.1·두 실험의 기본 off는 변경하지 않았다.
추가된 경로는 실행 예제, 진단 runner/검사, 해당 CPU CI뿐이다.

비교 출력은 다음 실제 값들을 포함한다.

- leaf 경로·full known history/FEN·origin/completeness/revision, 구체/공통 classification,
  ordered moves·order digest·semantic digest.
- C가 실제 request에서 만든 projection·dense tensor의 little-endian 바이트·input key·
  ordered policy indices. D 제출 직전 같은 불변 request로 prepare를 재현하고 D도 독립
  prepare 검증을 수행한다. 장치 input buffer를 직접 관측한 기록은 아니다.
- 실제 PUCT에 전달한 모든 edge 통계의 float bits·선택 index, accepted leaf value,
  selection별 backup 직후 root 통계·방문/backup 카운터, 최종 상태·best move·fallback.
- cancel/expiry/stale·node limit·잘못된 input key·비수치 output의 구체 오류,
  shutdown 후 실제 scheduler/physical lease·예약의 drain.

owner 발급 번호는 비교 출력에서 제외하지만 stale authority와 중복 소비를 정상화하여
숨기지 않는다. 향후 상태 재사용이 `play`를 생략하더라도 `snapshot`에서 선택 leaf를
갱신하도록 witness adapter를 구성했다. 출력 64MiB, visits 1~64,
`visits * 16 + 64` pump/탐색·30초/탐색·64 drain step 상한을 둔다.

[`paired_search.py`](../../benches/runtime/paired_search.py)는 별도의 실제 clock에서
UCI engine을 새 process로 직렬 실행한다. shell을 사용하지 않고 source/binary hash·
feature·compiler·명령·fixture·seed·자원 상한을 잠근다. 실제 합법 bestmove, stderr·
late duplicate·timeout·실패 exit·quit/drain을 검사하고 실패를 원본과 함께 보존한다.
native receipt는 PID/binary·model/encoding/backend·FP32/B1/worker/thread/revision·
물리 drain과 CUDA의 실행/mapping 증거를 요구한다. 이번에는 native 실행을 하지 않았다.
전체 스키마·명령·지원 범위는 [runtime 도구 문서](../../benches/runtime/README.md)에 있다.

### 10.2 원본·독립 feature matrix의 전체 출력 대조

기준 checkout에는 같은 witness 예제와 public trace fixture를 연결했다. 기존 probe의
example 선언 외 제품 소스 차이는 없으며 별도 baseline worktree의 선행 자료는 보존했다.
동일 Rust 1.96.0 release/locked/offline에서 original, current default, claim만,
history만, 두 옵션 조합의 5설정을 빌드했다. 각 설정에서 history fill `no`와 `always`를
별도로 실행했다. 정상 fixture 16개와 guard 6개, 정상 ongoing root당 visits 16이다.

입력군은 이력 1/17/65/129/257, unknown-prefix 시작 FEN, Kiwipete, castling·EP·
EP pin·승격/50수 intended claim, 현재/예정 3회·자동 5회 반복, mate/75수 우선순위,
최대 카운터 상태다. 최대 카운터 fixture는 자동 75수 terminal이 먼저 판정됐다.
**이 fixture가 overflow 오류를 발생시켰다고 보고하지 않는다.** 실제 checked transition의
overflow 원자성은 기존 `checked_counter_overflow_is_atomic`과 독립 A 검사로 검증한다.

각 fill의 witness 전체가 5설정에서 byte-for-byte 일치했다. `no` 출력에는
22 CASE·249 LEAF·226 TENSOR-LE·329 SELECT·222 ACCEPTED/BACKUP·22 FINAL/DRAIN이 있다.
root initialization과 completed visit/backup을 구분하며 terminal/guard를 16방문 완료로
세지 않는다. 출력과 binary별 hash는 `${ARTIFACT_ROOT}/opt00/matrix-05c6e5f/matrix.json`에 있다.

| history fill | 동일한 전체 witness SHA-256 |
|---|---|
| no | `2b1e60bf922f265b5b5d880608ea5876115b8bfa41243ca4f2e792e98c486599` |
| always | `42df4440743024ce8420381ec6472af9dd7e3bda51b6f1ca093c6aa5fc2ab86a` |

이는 실제 인코딩과 결정적 CPU/mock 탐색의 동등성 근거다. NN 수치 오차·GPU·통제 장비의
deadline 성능·대국 강도나 모든 체스 상태의 동등성 증명으로 확대하지 않는다.

### 10.3 실제 UCI 반복 실행: 별도 진단 cohort

실행 전에 profile off, CPU mock, seed 20261003, 동일 witness에서 추출한 trace-0/256
position·ordered legal moves, `go nodes 16`을 잠갔다. 원본→default,
default→claim, default→history, default→both를 별도 비교했다. 내부 API probe였던
8장 자료와 합산하지 않는다. 1차 지표는 **position 송신→bestmove 수신**이다.
이미 position 단계에서 수행되는 준비 비용을 포함하려고 pilot 전에 고정했으며,
3.4의 `go`→`bestmove`도 별도 지표로 보존한다. ready barrier·명령 전송은 양쪽에 동일하다.
startup/quit는 process wall에 별도로 남는다. mock의 기존 Waiting polling도 비용에 포함된다.

각 비교·fixture의 pilot은 control 3 + variant 3 block, 후속 비교는 3창×4 block,
각 block은 사전 배정한 ABBA/BAAB 4 fresh process다. 두 fixture를 합쳐 비교당
pilot 48·후속 96 process이며 총 **smoke 8 + pilot 192 + 후속 384 = 584 process**다.
후속 창은 같은 VM에서 연속 실행한 **개발용 3창**이며 통제된 독립 3세션으로 보지 않는다.
실행 시작 UTC는 17:23:35~17:25:11이다. 반복 수를 사후 축소하거나 실패·느린 표본을
대체하지 않았으며 584개 모두 diagnostic으로 완료했다.

자원 조건은 Linux 6.18.44, Xeon Platinum 8573C, Rust 1.96.0,
container quota 4 core/16GiB, engine CPU 0·runner CPU 4다. process timeout 12초,
run wall 60초, 관측 RSS 256MiB·run 산출물 64MiB·throttle 증가 허용치 0을 잠갔다.
해당 실행 동안 cgroup throttle·OOM/OOM-kill/high 증가가 없었고 최대 관측 VmHWM은
4,345,856 bytes였다. RSS는 Linux 샘플링 값이며 강제 OS limit 또는 모든 순간의
peak를 검증한 값으로 보지 않는다. CPU/memory PSI·host 독점·clock은 unknown,
GPU unavailable이다. 관측 증가가 없다는 사실로 외부 경합이 없었다고 확정하지 않는다.

현재 mock UCI는 `info nodes`를 출력하지 않아 **584개 모두 `reported_nodes=null`**이다.
`go nodes 16`은 요청 한도이며 실제 완료 방문 수로 대입하지 않았다. Rust witness에서
actual visit/backup을 검증했지만 다른 UCI 실행의 방문 처리량 증거로 재사용하지 않는다.
actual UCI counters 연결과 movetime/deadline 비교는 후속 검증 항목이다.

아래 값은 후속 12개 paired block의 평균 시간 B/A 비율 중앙값이다. 구간은 창 내부
block을 재표집한 10,000회 bootstrap의 진단용 95% percentile(seed 20261003)이며
독립 세션의 모집단 신뢰구간으로 인수하지 않는다. 1보다 작으면 해당 실행의 시간이 짧다.

| 비교 | 이력 1 비율 [진단 구간] | 이력 257 비율 [진단 구간] |
|---|---|---|
| original → default | 1.004 [0.984, 1.029] | 0.971 [0.955, 1.021] |
| default → claim | 0.989 [0.942, 1.003] | 0.936 [0.928, 0.959] |
| default → history | 0.995 [0.970, 1.007] | 0.767 [0.673, 0.806] |
| default → both | 0.983 [0.972, 1.012] | 0.775 [0.755, 0.803] |

baseline-baseline pilot도 보존했다. block 비율 범위는 짧은 이력 0.987~1.081,
긴 이력 0.909~1.053으로 흔들렸다. 비교별 전체 지표·control·원시 로그·UTC·affinity·
resource snapshot은 `${ARTIFACT_ROOT}/opt00/paired-05c6e5f/`에 있다. 긴 이력 비용 감소의
진단 근거는 얻었지만 2% 전체 탐색 회귀 허용폭·기본 활성화·GPU·강도는 인수하지 않는다.

### 10.4 검사·보존과 다음 작업

- 로컬 `cargo test -p rz-uci --example optimization_witness --locked`: 3 passed,
  0 failed. 실제 root 초기화·방문/backup, guard의 구체 결과와 physical drain·상한 검사.
- 로컬 Python runner 검사: 11 passed. 명시적 fake UCI controller로 hash/argv/순서,
  실제 nodes를 만들지 않음, illegal/duplicate/stderr/timeout/log 상한, native receipt/
  CUDA mapping 실패를 검사한다. fake controller 결과를 실제 engine/GPU로 표시하지 않는다.
- source `05c6e5f`의 [CI 37139504129](https://github.com/daejunnom/RoveZero/actions/runs/37139504129):
  Ubuntu·Windows의 fmt, workspace/all-target/all-feature, native CLI,
  release 독립 python-chess oracle/include-ignored, F/Python runner, strict Clippy
  필수 step 모두 성공을 직접 확인했다. 기존 미지원 ignored를 통과로 세지 않는다.

원시 matrix·binary·build log·witness, 모든 584 process·manifest·stdout/stderr·control과
진단 요약을 저장소 밖에 보존했다. 실행 후 저장한 `analyze.py`는 기존 표본을 재실행하거나
덮어쓰지 않고 모든 hash·manifest binding·출력 동일성을 재확인하여 `evidence-recheck.json`을
생성한다. 최초 요약과 재확인 source도 각각 유지한다. 공개 source snapshot·baseline overlay
차이·Cargo.lock hash·실제 검사/CI snapshot까지
`${ARTIFACT_ROOT}/opt00-evidence-20261003.tar.gz`에 묶었다. payload 1,295개와 내부 hash 목록
1개, 총 1,296개 member를 압축 파일에서 다시 읽어 전부 SHA-256 대조했다.
패키지는 12,766,385 bytes이며 SHA-256은 다음과 같다.

`c9d30797ad5766fbf39c0db9136ae9fcab6b231538905fbb4241e5177d34aaed`

회수 검증은 `${ARTIFACT_ROOT}/opt00/archive-verification.json`에 있다. 산출물은 이 작업의
cloud workspace 수명 동안 보존하며 자동 삭제하지 않는다. 회수용 패키지는 저장소 밖에
있고 원격 영구 보관은 아직 수행하지 않았다. 초기 개발 witness는 최종 matrix와 분리해
보존하며 현재 동일성/시간 표본에 합치지 않는다.

당시 다음 구현은 **OPT-01**이었다. 후속 전체 구현·검증은 11장에 기록했다.
비트보드 직접 읽기와 정확한 borrowed prefix 반복 계산을
독립 opt-in 변경으로 구현하고 이 witness로 전체 입력/고정 방문 trace를 다시 대조한다.
기존 두 feature와 새 변경을 섞어서 하나의 개선율로 보고하지 않는다. OPT-00의 CPU 기반은
완료했지만 통제 세션·UCI actual visits·observer off/on·실제 CPU/GPU 모델·강도·기본 활성화는
계속 미완료이며 A가 후속 구현·검증을 맡는다.

## 11. OPT-01~12 전체 구현과 검증

### 11.1 구현 범위와 활성화

사용자의 전체 A 배정에 따라 9장의 미반영 OPT-01~12를 모두 구현하고 A/C/D/B의
실제 소비 경계를 연결했다. 성능 행렬의 source는
`c73fa596d2a9651677055bc2706354d30f494f88`, 기본/전체 회귀 source는
`42d58682463080d2697181f970b8c2b8fe98acc4`다. 후자는 batch wakeup 테스트의
불필요한 변환을 제거하고 외부 수치 대조 예제에 실행 옵션을 연결한 후속이다.
두 소스 사이의 제품 library 변경은 `#[cfg(test)]` 블록에만 있다.
base는 `b5ba853585cb5a78f81159f733a86cdfe085936b`, PR #17
`81059e463af8afc151253b85306fc7c88e5b58bc`는 별도 미병합 참고 자료다.

| 단위 | 구현·실행 옵션 | 실제 연결·보존 조건 |
|---|---|---|
| OPT-01 | eval `experimental-bitboards`, `experimental-history-frames` | Rules bitboard 직접 읽기와 borrowed 전체-prefix 반복 계산을 독립 구현. 최근 8 frame만 반환하며 전체 규칙 이력·unknown-prefix·raw EP는 유지 |
| OPT-02 | eval `experimental-prepared-input` | 동일 history node의 weak identity와 정확한 ordered legal에 묶인 1-slot 불변 준비 결과. projection/tensor/key/indices를 공유하고 fresh request profile/budget/key 검증을 유지 |
| OPT-03 | eval `experimental-input-hash` | 256 float의 유한 stack buffer로 SHA update를 묶음. 기존 little-endian float bits·framing·input key 동일 |
| OPT-04 | search `experimental-puct`, `experimental-search-buffers`, uci `experimental-best-move` | streaming argmax, bounded 재사용 통계 buffer, 가벼운 진행 조회. f64 연산·첫 tie 선택·오류 우선순위·최종 outcome 유지 |
| OPT-05 | search `experimental-state-cache` | 전체 move path에 연결한 root 내 불변 Rules 상태를 가장 긴 prefix부터 재사용. entry/byte 한도·eviction·fresh authority 검증. opaque state의 크기를 모르면 miss |
| OPT-06 | uci/eval/runtime `experimental-raw-cache` + typed cache limits; native `--experimental-raw-cache` | game 내 root 간 전체 raw policy/WDL만 재사용. 실제 입력 bits·metadata·계산 profile 대조, D 최종 승인 후 승격, fresh admission/receipt, 실행 출처·Computed/hit 집계 분리, newgame reset |
| OPT-07 | uci/eval/runtime `experimental-notify` | worker publication 이후 완료 알림, authority 취소 알림, bounded deadline 대기. sequence를 pump 전에 읽음. queued native batch는 물리 worker가 없어도 200µs timer로 진행 |
| OPT-08 | position `experimental-attack-tables`, `experimental-pin-check` | knight/king/ray tables와 pin/check 필터를 독립 구현. 왕·castle·EP는 정밀 child 적용 유지. 합법 수 정렬·claim·counter 실패 보존 |
| OPT-09 | uci/eval `experimental-io-buffers` + `--experimental-io-buffers` | 같은 batch shape의 input tensor 1-slot, 최대 batch 수의 owned raw buffer pool. physical 종료와 ownership 반환 후 reuse. 보존 중인 출력은 수정하지 않음 |
| OPT-10 | uci/eval `experimental-io-binding` + `--experimental-io-binding` | fixed device input/output와 host 출력, synchronous ORT copy·Run·output fence. CPU 입력은 매번 bind. copy의 추가 Identity session 비용도 명시 |
| OPT-11 | uci/eval `experimental-cuda-graph` + `--experimental-cuda-graph` 및 explicit binding | CUDA B1·고정 shape/주소/session, EP Graph 옵션. 성공 binding call 수와 실제 capture/replay 관측을 구분. 불명확한 CUDA 완료는 session/input/binding quarantine |
| OPT-12 | uci/search/eval/runtime `experimental-batch` + native `--experimental-batch=N` | 폭 1..16 다중 ticket·virtual reservation, D mixed-legal capability, C 실제 multi-item single-worker batch. 완료 순서가 바뀔 수 있는 **S 실험** |

모든 실험은 기본 off다. compile flag만으로 raw cache·I/O mode·batch 폭을 runtime에서
켜지 않는다. 기존 claim-preview와 history-digest도 독립 off를 유지한다.
공통 revision 0.1·wire/입력/state codec은 유지했고 public Rust configuration 필드와
trait hook의 실제 호환 영향은 [계약 적용 9장](../CONTRACT-ADOPTION.md#9-pr-18-전체-최적화의-실제-소비-접점)에 기록했다.
native Computed-only V1 attestation은 raw-hit·실행 옵션·B>1과의 조합을 거부한다.

### 11.2 정확성·한도·수명 검사

`c73fa59`의 **16개 설정 × history fill no/always** 전체 witness가 원본과
byte-for-byte 일치했다. 설정은 original/default, claim/digest, bitboards/history-frames,
prepared/input-hash, puct/search-buffers/best-move/state-cache, attack-tables/pin-check,
combined-E, all-flags-serial이다. 마지막 설정은 모든 compile flag를 포함하되 raw
cache/실행 모드 off·S0 폭 1이며 GPU나 S1 실행으로 세지 않는다.
새 witness는 실제 Rules adapter의 retained-storage charge를 전달하므로 상태 cache가
opaque miss로만 끝나지 않는다. baseline의 이전 바이너리는 원래 hash를 검증하고 새
출력 루트에서 다시 실행했다. 예제 변경은 이 charge 전달뿐이며 비교 codec은 같다.

| fill | 출력 bytes / 설정 | 전체 출력 SHA-256 |
|---|---:|---|
| no | 16,004,271 | `2b1e60bf922f265b5b5d880608ea5876115b8bfa41243ca4f2e792e98c486599` |
| always | 16,006,835 | `42df4440743024ce8420381ec6472af9dd7e3bda51b6f1ca093c6aa5fc2ab86a` |

비교 대상은 10.1의 16 정상 fixture+6 guard, 실제 tensor bits/key/ordered indices,
전체 규칙 이력·classification/digest, selection/leaf/value/backup·종료/오류/drain이다.
이 증거는 같은 합성 평가와 고정 방문 수의 S0 동등성이며 neural/GPU나 모든 입력의
동등성 증명은 아니다. S1과 cache hit의 provenance는 별도 검사한다.

- 상태 cache의 zero/tiny/1-entry eviction/기본 한도에서 실제 완료 방문·root 통계·
  counter가 같고 보유량이 상한 안에 남는다. weak prepared cache의 독립 FEN miss,
  eviction 후 보존 tensor 불변성, 잘못된 profile/budget/key·legal 순서 거부도 검사했다.
- raw cache는 두 root 사이에서 fresh request/receipt를 반환하며 newgame은 Computed로
  돌아간다. 취소/만료 후 늦은 물리 완료는 entry/stage를 만들지 않고 실제 예약은 fence까지
  유지한다. 실제 A/C/D/B 종단 검사는 root별 **32 completed visits·초기화 1회**를 확인했다.
  폭 1의 같은 게임 root 2는 **33 RawEvalHit·새 물리 호출 0**이고 비대칭 WDL을 사용한
  방문/값/root 통계가 첫 root와 같다. raw cache가 NN 시간을 절약했다는 측정은 아니다.
- 실제 폭 4 종단 검사는 여러 입력의 한 physical batch, pending peak>1·batch peak>1,
  각 root의 33 accepted output과 32회 backup·예약 반환을 확인했다. 두 root의 총 물리
  호출이 B1의 66회보다 적다. legal 수가 다른 C/D batch에서 한 항목만 취소해도 다른
  항목은 완료하고 전체 physical fence 전에는 lease를 반환하지 않는다.
- B의 역순 완료·중복 완료·전체 취소는 각 ticket을 한 번만 소비/반환한다. virtual 통계는
  실제 방문을 미리 증가시키지 않는다. notifier는 wait 전/중 publication·취소·deadline을
  검사했고 queued native batch가 초기 완료 신호 없이 timer→worker→완료로 진행했다.
- CPU unit 경계에서 raw output pool의 실제 pointer 재사용·보존 출력 불변성·잘못된
  길이/비유한 값 거부와 실행 모드의 미지원·Graph 조건 거부를 확인했다. 실제 ORT input
  buffer/device address·Graph replay·quarantine의 장치 검사는 미실행이다.

`42d5868`의 직렬 로컬 검사 결과는 기본 workspace/all-target **620 passed, 16 ignored**,
전체 feature **695 passed, 16 ignored**, 기본 native CLI **4 passed**다. attack-only와
pin-only의 release/include-ignored는 각각 **45 passed**, 모든 A feature는 **50 passed**로
공개 depth-4 perft·고정 `python-chess==1.999`/`chess==1.11.2` 독립 대조와 doc test를
포함한다. fmt와 strict Clippy도 통과했다. Python runtime runner는 **11 passed**, 기존
model tool 회귀는 **105 passed**다. ignored는 통과로 세지 않는다.

첫 전체 회귀의 Debug 길이 제한 실패와 첫 Clippy 실패, `c73fa59` CI의 새 테스트
동일 타입 변환 Clippy 실패는 원시 로그에 보존했다. 이후 수정·검사 결과와 합치거나
실패 표본을 지우지 않는다. 실제 검사 명령·source·log hash는
`${ARTIFACT_ROOT}/allopt/validation-42d5868.json`과 `logs/`에 있다.

### 11.3 재현·장치 검사와 남은 인수

빌드는 Rust 1.96.0, release, locked/offline, `-j4`이며 산출물은 저장소 밖에 둔다.
독립 옵션마다 다음 빌드/실행을 수행하고 즉시 binary hash·feature·source를 고정한다.

```sh
CARGO_TARGET_DIR="$RZ_OPT_OUTPUT_ROOT/build" cargo build --release --locked --offline -j4 \
  -p rz-uci --bin rz-uci --example optimization_witness --features FEATURES
"$RZ_OPT_OUTPUT_ROOT/build/release/examples/optimization_witness" \
  --visits 16 --history-fill no
"$RZ_OPT_OUTPUT_ROOT/build/release/examples/optimization_witness" \
  --visits 16 --history-fill always
cargo test --workspace --all-targets --all-features --locked
cargo test --release -p rz-position --all-features --locked -- --include-ignored
```

여기서 `FEATURES`는 위 표의 crate prefix를 포함한 정확한 feature 목록이며
`${ARTIFACT_ROOT}/allopt/matrix-c73fa59/matrix.json`에 각 build argv·binary/witness
hash를 고정했다. CPU/mock runner의 명령·상한은 [runtime README](../../benches/runtime/README.md)를 따른다.

`maia_check`는 기존 외부 원본 protobuf·ONNX·pinned ORT·독립 LC0 fixture·report
인자 뒤에 `--experimental-io-buffers`, `--experimental-io-binding`,
`--experimental-cuda-graph`를 명시해 새 경로의 수치를 대조할 수 있다. Graph는 CUDA와
explicit binding이 필수이고 B1만 실행하며 2/4/8/16 제외를 보고서에 남긴다.
실험마다 같은 세션/shape의 32회 B1 반복과 보존 출력 불변성, host staging·전송·Run·
output fence·출력 소유화 구간을 기록한다. 시계는 CPU wall clock이며 GPU event가
아니다. `binding_runs`나 반복 수를 실제 capture/replay의 관측으로 쓰지 않는다.

현재 host에는 GPU와 이 검사용 외부 자산·pinned ORT bundle이 없어 새 실제 CPU NN 및
GPU 수치·장치 수명·capture/replay·전송/VRAM/성능 인수는 **미실행**이다. 과거 C의 CPU
수치 인수를 새 실행 옵션의 검증으로 재사용하지 않는다. 후속 지정 장비 검증은 baseline,
buffers, binding, fixed B1 Graph, S1 batch를 분리하고 새 backend identity와 모든 자기
시간·warm-up·peak memory를 기록한다. Graph와 S1 확대는 한 variant로 합치지 않는다.
동일 W·CPU/GPU/시간/메모리 holdout 대국, observer off/on, UCI actual visits와 통제된
독립 세션·기본 활성화 판단도 남아 있다. 구현과 소비자 연결을 다른 담당에게 넘기지
않으며 새 유료 자원을 만들거나 성과를 추정하지 않는다.

### 11.4 새 직렬 CPU 진단 cohort

`c73fa59`의 immutable binary를 사용하여 2026-10-03 **20:29:23~20:31:02 UTC**에
새 **2,128 process**를 실행했다. smoke 112 + pilot 672 + confirm 1,344회다.
original→default 1개, default→12개 독립 옵션과 combined-E의 13개로 총 14 비교다.
각 비교는 짧은/긴 두 이력 fixture, baseline-baseline pilot, ABBA/BAAB, 3개의 직렬
개발창 × fixture별 4 confirm block이다. 창은 같은 VM의 연속 실행이며 통제된 독립
세션이 아니다. 8·10장의 이전 표본과 합산하지 않았다.

빌드·검사는 계측 전에 끝냈고 다른 local build/test와 겹치지 않았다. Rust 1.96.0,
release/locked/offline, mock provider·profile off·PR #17 observer 미병합, engine CPU 0,
runner CPU 4, quota `400000/100000`, 16GiB cgroup을 고정했다. host 독점·CPU clock·
PSI는 unknown이며 GPU/NN을 실행하지 않았다. 모든 process의 stdout/stderr·binary/
source/fixture/manifest hash를 재대조했다. cgroup throttle·OOM/oom_kill/high 증가는
관측상 0이고 진단 분류는 2,128회다. 이것으로 host의 다른 경합을 배제하지 않는다.
샘플링한 Linux `VmHWM` 최대는 **4.180MiB**이며 강제 RAM 상한·device memory가 아니다.

1차 지표는 position 송신→bestmove 수신이며 position 준비·ready barrier를 포함한다.
go 구간·process wall도 원시 기록에 따로 남겼다. `reported_nodes`는 2,128회 모두
null이다. 요청한 `go nodes 16`을 관측한 완료 방문 수나 NPS로 바꾸지 않는다.
실제 방문·backup의 정확성은 별도 Rust witness/주입 종단 검사에서 확인했다.

아래는 12개의 confirm paired block 평균 B/A 비율의 중앙값이다. 작을수록 해당
fixture의 전체 지연이 짧다. 이전 source와 서로 다른 옵션의 효과를 합치지 않는다.

| 비교 B/A | 이력 1 | 이력 257 |
|---|---:|---:|
| original → default | 0.996 | 1.014 |
| default → claim | 0.982 | 0.960 |
| default → digest | 0.985 | 0.793 |
| default → bitboards | 0.991 | 0.993 |
| default → history-frames | 1.004 | 0.912 |
| default → prepared | 0.967 | 0.959 |
| default → input-hash | 0.999 | 1.004 |
| default → puct | 1.008 | 0.974 |
| default → search-buffers | 0.995 | 0.988 |
| default → best-move | 1.005 | 0.998 |
| default → state-cache | 1.006 | 0.999 |
| default → attack-tables | 0.998 | 0.928 |
| default → pin-check | 0.998 | 0.957 |
| default → combined-E | 0.943 | 0.661 |

combined-E의 창 내 block bootstrap 95% 진단 구간은 이력 1 `[0.911, 0.960]`,
이력 257 `[0.622, 0.691]`이다. seed 20261003, 10,000 draws, 직렬 개발창별 4 block을
재추출한 중앙값의 nearest-rank percentile이다. 독립 세션의 불확실성으로 해석하지
않는다. baseline-baseline control 중앙값 범위는 `[0.968, 1.039]`, 개별 control block은
`[0.815, 1.217]`로 흔들렸으며 느린 표본을 제거하거나 재시도하지 않았다.

긴 이력의 digest·history frame·입력 준비·공격 테이블과 전체 조합에서 비용 감소 신호를
확인했다. bitboard/hash/buffer/state-cache 등의 개별 신호는 이 fixture에서 작거나
불확정이다. puct와 best-move의 짧은 이력에는 약 0.8%/0.5% 증가 신호도 남겼다.
original→default의 긴 이력 구간 `[0.995, 1.041]`은 2% 회귀 허용폭을 넘으므로 기본
경로의 정식 무회귀 인수도 통과로 표시하지 않는다. 모든 독립 구간·창별 block·control·
process 중앙값은 `${ARTIFACT_ROOT}/allopt/evidence-recheck.json`에 있다.

이는 **CPU/mock·두 fixture의 진단**이다. 34%의 긴 이력 조합 신호를 NN/GPU나 대국
강도·모든 국면으로 확대하지 않는다. raw cache/notify/I/O/batch의 활성 runtime 성능도
이 비교에서 측정하지 않았다. 그 경계는 위 정확성 검사와 별도 장치/품질 인수로 남긴다.
계측 후 `a5c4c10`은 S1 virtual selection에서 버려지던 원본 통계 수집을 제거하고,
search-buffers 옵션에서 virtual 통계와 index mapping도 재사용하도록 개선했다.
이 후속의 S1 성능을 `c73fa59` 표본에 포함하지 않는다.

### 11.5 최종 source 재검사와 보존

최종 제품 source `a5c4c10ee8b4ccb07c65a7b5228b14622b2c9d3a`에서 기본·combined-E·
all-flags-serial을 다시 release 빌드하여 두 history fill의 전체 witness hash가 위 원본과
같음을 확인했다. 최종 S1 buffer 정리 이후의 search 전체 feature 및 실제 A/C/D/B
다중 요청/캐시 종단 검사도 통과했다. 전체 workspace/all-target/all-feature는 다시
**695 passed, 16 ignored**, 기본 설정도 **620 passed, 16 ignored**다.
이전 A 독립 oracle/Python 도구 검사와 해당 소스 차이는
실제 영향 범위를 기준으로 구분한다.

[코드 source `a5c4c10` CI 37152168784](https://github.com/daejunnom/RoveZero/actions/runs/37152168784)의
Ubuntu·Windows에서 fmt, workspace/all-target/all-feature, native CLI,
release/include-ignored 독립 python-chess oracle, Python model/runtime 도구, strict
Clippy 필수 step이 모두 성공했음을 직접 확인했다. `42d5868`의
[CI 37150906644](https://github.com/daejunnom/RoveZero/actions/runs/37150906644) 성공과
`c73fa59`의 [CI Clippy 실패](https://github.com/daejunnom/RoveZero/actions/runs/37150250423)는
각 source의 결과로 보존한다. CI 요청·ignored·GPU 미실행을 통과로 바꾸지 않는다.

새 자료는 `${ARTIFACT_ROOT}/allopt/`에 별도 보존했다. `matrix-c73fa59/`,
`final-witness-a5c4c10/`, `paired-c73fa59/`에 binary/build argv·witness·모든 process와
stdout/stderr·manifest/control을 유지한다. `build_matrix.py`, `verify_final_witness.py`,
`validate.py`, `paired_diagnostics.py`, `analyze.py`와 원시 실패/성공 log도 보존한다.
기존 8·10장 evidence와 archive는 수정·삭제하지 않았다.

회수용 `allopt-evidence-20261003.tar.gz`는 **50,944,423 bytes**, payload 4,810개와
내부 hash 목록 1개로 총 **4,811 member**다. 압축본에서 모든 member의 SHA-256을
재대조했다. archive SHA-256은 다음과 같다.

`a3892c5c69c8171aebd0ab751a306ebcf303a2eb10d805cb47004af71648c82b`

이 패키지는 코드 `a5c4c10`·계측 `c73fa59`·문서 `b81625d` snapshot이며 자신의
checksum을 기록하는 후속 문서보다 먼저 생성했다. cloud workspace 수명 동안
보존하며 자동 삭제하지 않는다. 원격 영구 보관은 아직 수행하지 않았다.

이후 테스트 전용 보강으로 폭 4+raw cache를 함께 켜 두 root를 검사했다. root 2의
non-root cache backup도 실제 발생하며 32 completed visits·33 accepted output·
33 reservation release·pending/resource 0을 유지한다. S1의 루트 통계를 S0와 같다고
assert하지 않는다. 추가 테스트·후속 문서·최신 CI snapshot은 기본 보존 패키지와
SHA로 연결한 별도 source 보존본에 남긴다. 이 보강은 제품 library와 기존 계측 표본을
변경하지 않았으며 실제 NN·GPU의 batch/캐시 수치 대조는 여전히 미실행이다.

## 12. PR #17·#18 수동 통합과 병합 준비

2026-10-04 총괄 TASK-I02는 open/draft 상태의 #17과 #18, develop
`b5ba853585cb5a78f81159f733a86cdfe085936b`, 실제 CI·리뷰와 두 작업 checkout을 확인했다.
외부 review/review thread는 없었다. #17 source `81059e4`와 #18 source `278b146`의
여섯 충돌 파일을 수동 연결하고, 후속 수정까지 합친 제품 소스
`a93569bedb802a4eb07f19b12241595715d21df1`을 검사했다. #18에 #17 ancestry를
보존했으며 공통 계약 revision은 **0.1**이다. 후속 문서 head는 제품 검사 소스와 구별한다.

### 12.1 병합 전 해결한 실행·계측 오류

- C worker·native owner·ORT Run timing·buffer 반환과 B/D/UCI의 profile/notification/
  batch/backup 접점을 수동으로 맞췄다. D02 v1은 fresh Computed B1 전용이므로
  `--profile`+raw cache/폭 2 이상/buffer/binding/Graph를 asset 로딩 전에 거부하고
  C의 profiled worker/backend에서도 방어한다. raw-hit aggregate를 complete Computed
  timeline으로 인수하지 않는다. 별도 cache/batch/device 계측이 필요하다.
- OPT-00 native runner의 manifest pair가 Rust의 `--flag=value` CLI와 맞지 않던
  오류를 실제 Popen 경계에서 수정했다. 공백·등호 유지, 누락/중복 flag 거부를 검사했다.
- 첫 실제 CPU profile 실행 `673f3e4`는 두 합법 착수·physical drain 뒤 quit 10초
  한도를 넘겼다. 직접 파일로 전달하던 작은 JSON write가 Windows mount에서 지연되어
  387,606 bytes의 부분 JSON을 남겼다. #17 `b830121`은 producer 종료 후 64KiB 버퍼로
  저장하며 8MiB cap·명시적 flush·sync·실패 전달을 유지한다. 기존 실패/부분 JSON은
  보존했다. 한도 초과·flush 실패와 실제 write 합치기·완전 JSON 검사를 추가했다.

### 12.2 현재 소스의 정확성·소비자 인수

| 검사 | 제품 소스·범위 | 실제 결과 |
|---|---|---|
| #17 CI | `b830121`, [run 37158044723](https://github.com/daejunnom/RoveZero/actions/runs/37158044723) | Ubuntu workspace 683 passed·16 ignored, Windows 629 passed·2 ignored; 실패 0 |
| #18 통합 CI | `a93569b`, [run 37158070152](https://github.com/daejunnom/RoveZero/actions/runs/37158070152) | Ubuntu workspace 705 passed·16 ignored, Windows 651 passed·2 ignored; 실패 0 |
| 별도 필수 CI step | 위 두 OS | fmt·기본 native CLI·release independent Rules oracle·Python model·strict Clippy 성공. #18 runtime runner 12개도 성공 |
| C 실제 CPU NN | `673f3e4`, ORT 1.22.0 CPU FP32, default/buffers/binding/buffers+binding | 각 12 독립 참조·B1/2/4/8/16 대조 passed. 비기본 모드는 32 반복 B1·보존 출력 불변도 passed |
| C worker·contract | 위 수치 gate의 fixture Rules view | 물리 출력 4개·취소 거부 1개; D scheduler를 포함한 검사는 아님 |
| full serial witness | `a93569b` all-feature serial, fill `no`와 `always` | 원본 전체 출력 hash와 동일; 각각 16,004,271 / 16,006,835 bytes |

로컬 통합 직전 `be7004b`의 Rust workspace 704개·기본 native CLI 4개·release
Rules/perft/oracle 50개·Python model 105개도 통과했다. runner 수정 뒤 12개, 통합 저장
수정 뒤 profile 관련 7개를 새로 실행했다. 최신 소스의 full 결과는 위 실제 CI이며
앞선 실행과 합산하지 않는다. 각 OS의 cfg별 검사 수 차이와 ignored는 그대로 남긴다.

C 수치 결과는 `maia_check` binary
`6f14806cab55b3102f91e5b40c698aa95ecf68a1c63958d2af519a4d902a412a`, export manifest
`63e4f9c28f2ce798c0f241583bbb338bffb89ad058ccf7117d3813eff441a0c9`, ORT library
`7520bc1b4f649ee8fa956442a66ded303e8fd3b465b0159b7e893bdc87df4e38`에 묶었다.
`673f3e4 → a93569b` 사이의 Cargo 파일·toolchain·contracts/eval/encoding source가 같고,
동일 환경·assets·binary·report를 확인하여 재사용했다. 새 NN 수치 실행으로 보고하지 않는다.
full witness의 SHA-256은 fill `no`가 `2b1e60bf922f265b5b5d880608ea5876115b8bfa41243ca4f2e792e98c486599`,
`always`가 `42df4440743024ce8420381ec6472af9dd7e3bda51b6f1ca093c6aa5fc2ab86a`다.

### 12.3 실제 CPU UCI·native runner와 종료 재검사

통합 소스의 새 release `rz-uci` SHA-256은
`00f2b269ed842a164d6e8f3fe1d3ad8e344c0775976c9b7dd50821018dacd13e`다.
ORT CPU·FP32·worker/thread 1, startup 45초·착수 20초·quit 10초·전체 90초,
각 stdout/stderr 2MiB·queue 256·line 64KiB 한도를 고정했다. 각 root의 명령은
`go movetime 10000 nodes 32`다. stdout에 없는 완료 방문 수를 요청 nodes로 대체하지 않는다.

| fresh 프로세스 모드 | root·관측 | 결과 |
|---|---|---|
| 기본 +profile/+attestation | startpos / e2e4 이후, e2e4·c7c5, fresh Computed 66 | complete journal/timeline, physical 시도·완료·전달·소비 각 66, 오류·미소비 0, confirmed drain·exit 0 |
| raw cache | 같은 startpos 두 번, ucinewgame 뒤 한 번 | 합법 착수 3개·fresh Computed 66·exit 0; 새 게임 초기화 확인 |
| I/O Binding | startpos / e2e4 이후 | 합법 착수 2개·fresh Computed 66·exit 0 |
| 폭 4 | startpos / e2e4 이후 | 합법 착수 2개·fresh Computed 66·exit 0 |
| 폭 4+raw cache | 같은 startpos 두 번 | 합법 착수 2개·fresh Computed 35·exit 0; S0와 동일 방문 분포를 주장하지 않음 |

다섯 모드 모두 native fatal/overflow/boundary/poison aggregate는 0이며 중복 bestmove가
없었다. 기본 profile의 quit→exit는 **0.063678초**였다. 원시 journal/termination hash와
실제 명령·PID·출력 hash·시간은 별도 receipt에 보존했다. raw/cache/batch 실험은 V1
attestation을 요청하지 않았다. fresh Computed 계수만으로 전체 cache hit나 batch 실행
분포를 만들지 않는다. 이 검사는 통제된 성능 A/B·GPU 개선·대국 강도 결과가 아니다.

수정한 production `paired_search.run_process`도 기본 CPU 모드를 새 프로세스로
실행했다. source·binary·model·encoding·backend identity와 V1 startup/termination을
대조했고 실제 Computed 17개·합법 `e2e4`·confirmed drain을 기록했다. 분류는
**diagnostic**, `formal_acceptance=false`다. 준비 스크립트의 첫 경로 선점 실패와 두 번째
기존 자료까지 포함한 64MiB cap 실패도 보존했다. 세 번째는 `run()`과 같은 전용 cohort
parent를 사용하여 동일 cap으로 통과했다. 느린 표본을 대체한 성능 cohort가 아니다.

### 12.4 원시 보존·병합 방식·남은 인수

Windows 저장소 밖 `reports/coordinator-integration/pr17-pr18-merge-20261004/`에
`cpu-numerical-receipts.json`, 실패 `cpu-uci-receipts.json`, 새
`cpu-uci-v2-receipts.json`, `cpu-runner-native-cohort-v3/`, 두 CI log와 full witness,
원래 stdout/stderr·부분 JSON·실제 명령 recipe를 보존한다. build/native library/weights는
공유 문서에 넣지 않는다. 후속 문서 변경은 영향 source·환경 일치를 확인한 재사용이다.

준비한 코드는 **#17 → #18 순서로 develop에 Merge commit**을 권고한다. #18에 #17
ancestry를 포함했으므로 첫 PR 이후 추가 변경만 검토할 수 있다. 현재 기록은 병합
준비이며 실제 develop/main 변경은 실행하지 않았다. 새 I/O/batch/cache의 CUDA 수치·
수명·VRAM, Graph capture/replay, RunPod 통제 A/B, S1 holdout 대국 품질, 실제 학습·
정식 LC0 paired 강도·통계는 남아 있다. 기존 GPU 인수나 CPU 진단으로 승격하지 않는다.
