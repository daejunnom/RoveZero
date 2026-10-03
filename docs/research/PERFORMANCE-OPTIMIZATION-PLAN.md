# 중복 연산·자료 복제·탐색 병목 개선 계획

작성일: 2026-10-03 UTC. 작성 담당: A(Codex). 확인한 공개 GitHub 작성자:
[daejunnom](https://github.com/daejunnom). 상태: **계획 및 A 실험 구현·CPU 진단; 기본 비활성**.

이 계획은 현재 소스 조사와 PR #17 재검토에 다음 사용자 지시를 반영한다.

- 기존 계측은 후보 선정의 참고 자료로 사용하고, 새 변경의 성능은 다시 측정한다.
- 자원 경합과 단일 실행의 오염 가능성을 통제하며 여러 차례 비교한다.
- 탐색 의미 또는 전체 실행 흐름이 달라질 수 있는 후보도 실험 코드로 구현한다.
  해당 PR에 실험임을 밝히고 변경에 맞는 테스트 항목을 추가한다.

A의 직접 구현은 `crates/rz-position/`이다. B/C/D/E 접점은 각 담당과 총괄이 연결할
구현 제안으로 구분한다. 공통 계약의 소유·E/A/S 분류·대국 검증은
[개발 기준](../ENGINEERING-STANDARDS.md), [실험 계약](../EXPERIMENTS.md),
[공통 계약](../CONTRACTS.md)을 따른다. 여기서 E/A/S의 A는 근사·모델 변경 분류이며
작성 담당 A와 다르다. 1~7장은 초기 계획·조사 이력이며, 실제 A 구현·검사·반복 CPU
진단은 8장에 기록한다. 정식 탐색 성능·GPU·대국 강도 인수는 보류 상태다.

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

아래 위치는 PR #17 head를 조사한 결과다. 실제 구현 전 최신 통합 SHA에서 존재 여부를
다시 확인한다. 우선순위는 가설이며 성능 개선율을 보장하지 않는다.

| 순서·담당 | 소스와 비용 구조 | 구현 후보 | 예상 분류·실험 표시 |
|---|---|---|---|
| 1 · A | `position.rs` 반복 횟수·완전성 별도 순회, `outcome.rs`의 모든 intended claim용 임시 child/history/owner | 반복 증거를 한 번에 계산하고 checked transition 결과만으로 claim 검사; 실제 child를 만들 때만 소유 객체 구성 | E 후보; claim 경로 변경의 동등성 검사 |
| 2 · A | `contracts.rs::state_digest`가 export마다 전체 이력을 FEN으로 변환·해시 | 직렬화 버퍼 재사용, 불변 상태와 profile에 맞는 digest 재사용 | E 후보; memoization이 실행 경로를 바꾸면 실험 표시 |
| 3 · A/C | `rules_projection.rs::project`가 보유 비트보드를 64칸에서 재구성하고 각 최근 frame의 전체 과거를 재순회 | 읽기 전용 비트보드·정확한 반복 metadata 제공 및 재사용 | E 후보 + 실험 표시; 모델 입력 바이트 대조 |
| 4 · C | `input_key`와 `prepare`가 projection·dense encoding·hash·legal index 작업을 반복 | C가 생성·검증한 불변 준비 결과 재사용; 같은 little-endian 바이트를 묶어 hash update | E 후보; 준비 결과 공유는 실험 표시 |
| 5 · B | `ContractSearch::pump`가 매 selection마다 루트부터 이동을 재생성하고 각 child를 export | root 수명 안에서 노드별 검증된 불변 상태를 유한한 용량으로 재사용 | E 가설 + 실험 표시; 선택 순서가 달라지면 S |
| 6 · B | selection의 `Vec<EdgeStats>`, PUCT의 점수 Vec, best move 조회용 전체 outcome·문자열 생성 | 작업 버퍼 재사용, 점수 생성과 argmax 결합, 가벼운 조회 API | E 후보; f64 연산·tie·오류 순서 유지 |
| 7 · C/D/B | computed-only 경로에서 동일 입력을 다시 추론 | 같은 game의 root 사이 bounded exact raw-eval cache | E 가설 + 실험 표시; provenance·receipt·소비 경로 변경 |
| 8 · B/D | Waiting에서 1ms sleep 후 poll | 완료·취소 통지와 deadline을 함께 처리하는 유한 대기 | 실험 표시; 고정 방문 trace를 비교하여 E/S 판정 |
| 9 · A | 모든 의사 합법 수에 state clone/apply/attack 검사, 기하 계산 반복 | 공격 테이블, pin/check/evasion mask, 예외 수의 정밀 검사 | E 후보 + 실험 표시; 광범위 movegen 경로 변경 |
| 10 · C/D/B | 매 Run의 staging Vec·출력 복제, B1 단일 진행 요청 | buffer 재사용 → I/O Binding → 별도 CUDA Graph·다중 요청/batch 실험 | 앞 두 항목은 E 가설 + 실험 표시; 선택·스케줄 변경은 S |

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

| 단계 | 구현 산출물과 책임 | 다음 단계의 검증 근거 |
|---|---|---|
| 측정 준비 | A의 기존 public 호출 probe와 D/E의 runner·manifest를 연결; 위 반복·오염·off/on 조건 구현 | 작은 smoke와 오염 표시/실패 분류 확인; 기존 자료는 참고로 보존 |
| A 내부 개선 | 반복 증거 단일 순회 → claim 임시 객체 축소 → 직렬화/digest 재사용을 각각 작은 변경으로 구현 | 규칙·오류·identity witness 일치, 각 변경의 반복 CPU 비교 |
| 소비 경계 개선 | A 읽기 전용 metadata와 C 준비 결과 재사용, B 통계 할당 제거·노드 상태 재사용을 독립 변경으로 구현 | 입력 바이트·fixed-visit trace·수명·메모리 상한 검사 |
| runtime 실험 | PR #17의 후속 후보인 exact raw cache를 먼저 구현; 완료 통지 개선은 별도 비교 | computed/cache provenance·receipt·exactly-once 소비·reset·취소 검사 |
| 확장 실험 | pin/check movegen, buffer/I/O Binding, CUDA Graph, 다중 요청/batch를 각각 분리 | 해당 정확성·흐름·장치 검사; S/A 후보는 별도 탐색·수치·강도 검증 |

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
처리한다. 새 옵션·공통 필드·revision과 통합 인수는 해당 담당 및 총괄이 맞춘다.

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
- 활성화 옵션·기본값·중단/복귀 방법, 영향 담당·계약 revision:
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
실험 구현·새 반복 CPU 진단은 아래에 추가한다. 나머지 담당 후보, controlled 전체 탐색·
GPU A/B, 실제 강도 인수는 후속 항목이다.

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
실제 모델·GPU와 동일 자원의 대국 강도다. B/C/D 후보 및 A movegen/projection 후보는
이번 구현 범위에 포함하지 않는다. 기본 활성화 전에는 allocation 감소와 짧은 API
개선만으로 이를 대체하지 않고 각 담당·총괄이 같은 integration SHA에서 확인한다.
