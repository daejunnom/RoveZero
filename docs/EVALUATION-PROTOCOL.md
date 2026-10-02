# 평가 기반 구현 지시서

이 문서는 [v0.2.0 핸드오프](reference/RoveZero_Handoff_v0.2.0_KO.md)의 2장,
11~12장, 14장, 17장을 R0 기준선·R1 계측·최종 대국의 구현 작업으로 풀어 쓴다.
연구 순서·승격 원칙은 [EXPERIMENTS](EXPERIMENTS.md), 상태·요청 타입은
[CONTRACTS](CONTRACTS.md), 미결정 값은 [DECISIONS](DECISIONS.md)를 따른다.
아래 스키마와 검사 이름은 구현할 계약이며 현재 존재하는 명령·검사기·실행 결과가 아니다.

현재 사용자는 **독립 Rust workspace와 자체 MIT 코드, 외부 가중치의 권리 분리**를
선택했다. 총괄로 배정된 에이전트가 최소 공통 계약을 직접 정하고 `rz-contracts`의
버전·변경을 소유한다. TASK-A03은 상태·평가 계약을 적용·검증하는 구현 작업이며
공통 계약의 독립 결정권자가 아니다. 문서 상세화가 실제 Cargo 생성·구현·실험
실행을 완료했다는 뜻은 아니다.

A~F의 클라우드 개발 환경에는 GPU가 없을 수 있다. fixture·CPU/mock와 가능한 CPU
신경망 참조, CPU 외부 엔진 runner로 manifest·pair·PGN·실패·통계·검사 진입점을
개발한다. D02의 CPU/mock 지연은 실제 GPU 지연과 별도 결과이며 D03의 GPU 개선을
입증하지 않는다. CUDA capability 부재와 GPU 검사 skip·미실행을 명시하고 GPU
모드를 조용히 CPU로 대체하지 않는다. 정식 GPU 강도 비교는 지원 환경의 고정
장비·backend·자원·시계에서 총괄 I02가 실제 GPU·paired 대국 근거를 확인한 뒤 인수한다.
이 문서 공유는 새 실제 대국·실험 또는 유료 GPU 사용을 승인하지 않는다.

## 1. 산출물과 선행 결정

평가 기반의 산출물은 잠긴 manifest, complete-state opening 묶음, pair 계획,
러너 어댑터, 독립 결과 감사기, 원시 사건 기록, 통계 보고다. 검증된 기존 러너와 통계
구현을 우선 비교하고 필요한 연결부만 구현한다. 평가 대상 엔진 내부의 자체 판정이나
출력 숫자만으로 정확성·시간 공정성을 확정하지 않는다.

| 결정 | 실행 전에 필요한 값 | 미정일 때 가능한 작업과 제한 |
|---|---|---|
| RZ-O002/O003 | 독립 Rust/MIT에서 쓸 외부 구성요소의 권리, 실제 가중치 형식·backend·러너 | 독립 Rust/MIT의 사용자 결정에 맞춰 계약을 구현한다. 외부 코드·가중치 권리와 추론 backend의 선택은 별도 잠근다. |
| RZ-O005 | 기준 LC0 source commit·바이너리·가중치·호환 backend·검색·정밀도·옵션 | manifest 필드와 오류 계약 구현 가능. 기준선 대국·속도 비교는 잠금 후 진행한다. |
| RZ-O006 | CPU/GPU·RAM/VRAM·전력·worker·wall-time·저장 상한 | 자원 회계 구현 가능. 무제한 pilot·멀티 GPU 실행은 시작하지 않는다. |
| RZ-O007 | opening·규칙/claim·tablebase·시계·표본·검정·중단·승격 정책 | pair 감사와 실패 분류 구현 가능. 미결정 값을 임의 채운 강도 실험은 시작하지 않는다. |
| RZ-O009 | 입력 이력·WDL 관점·정책 순서·정밀도별 오차 | 형식 독립 계약 구현 가능. 호환성이 확인되지 않은 evaluator는 연결하지 않는다. |

실행 한도에는 최대 game/pair, 최대 실제 경과 시간, worker와 자식 process 수,
출력 바이트, 취소·강제 종료까지의 유한 유예 시간을 포함한다. 예산 끝에서 결론이
나지 않으면 `inconclusive`로 남긴다. 기존 데이터를 유리하게 선별하거나 예산을
조용히 늘리지 않는다.

### 1.1 내부 대조와 외부 기준선

평가 기반은 아래 두 비교를 모두 지원한다. 내부 대조는 새 탐색·가중치의 원인을
분리하고, 외부 LC0는 최종 연구 목표를 확인한다. 같은 weights를 사용해도 RoveZero의
자체 Rust baseline PUCT가 LC0 구현·검색·강도와 같다고 주장하지 않는다.

| 비교 | 고정할 것 | 비교할 것 |
|---|---|---|
| 내부 CONTROL-0→CONTROL-1 | 같은 원본 weights·입력·후처리·GPU backend·자원·시계 | CONTROL-0 자체 Rust baseline PUCT → CONTROL-1 자체 새 search. 학습 weights 변경 없이 탐색 효과를 검증한다. |
| 내부 CONTROL-1→CONTROL-2 | 같은 CONTROL-1 새 search·입력·backend·자원·시계 | CONTROL-1 원본 weights → CONTROL-2 TASK-F02 fine-tuned weights. 탐색 변경 효과와 섞지 않는다. |
| 구조 비교 | 동일 데이터·학습/튜닝 예산·출력 계약·고정 search | TASK-F03에서 단일 architecture 변경과 대응 기준선을 비교한다. 형식·입력이 바뀌면 별도 모델군이다. |
| 외부 목표 | LC0 identity와 양쪽 전체 장비·시간 상한 | 실제 RoveZero 엔진 ↔ 외부 UCI LC0. 내부 소형 기준선 개선으로 외부 강도 향상을 대신하지 않는다. |

TASK-C02의 가중치 선정·단일 형식과 TASK-C03의 실제 NN 추론은 내부 대조의
선행 조건이다. CPU mock, 실제 CPU reference, 같은 weights의 GPU 출력 대조,
자체 Rust 엔진 연결, paired arena를 순서대로 확인한다. mock·상수/난수 evaluator,
외부 UCI LC0 연결만으로 TASK-C03의 내부 NN 또는 자체 엔진을 구현했다고 표시하지
않는다. TASK-D02의 자체 같은-weights 실행 계측 뒤 TASK-D03은 한 실행 최적화를
비교하고, search 의미를 바꾼 실험은 별도 내부 CONTROL-0→CONTROL-1 대조로 기록한다.

`TASK-C02`는 작업 ID, `CARD-C02`는 핸드오프의 recurrent 후보 ID다.
두 namespace를 문서·manifest·보고서에서 구별한다. 전체 작업 소유·의존성은
[IMPLEMENTATION-DIRECTIVES](IMPLEMENTATION-DIRECTIVES.md)를 따른다.
첫 weights는 권리·RTX 4050 Laptop 6GB 자료상 적합성 조사를 거쳐
[WEIGHT-SELECTION](WEIGHT-SELECTION.md)의 Maia1 v1.0으로 선정했다. 이 조건은 실행 시계·
학습 예산을 대신하지 않으며, CONTROL-0/CONTROL-1/CONTROL-2에 서로 다른 weights family를 쓰면
순수 search/미세조정 효과가 아닌 비교라고 명시한다.

## 2. manifest와 결과 스키마

machine-readable manifest를 먼저 구현한다. JSON/YAML 중 형식은 스택 결정 후
정하되 schema version과 단위는 필수다. 아래는 필요한 필드의 설계 목록이며 배포된
config 파일이나 유효한 예제 설정이 아니다. 미정·placeholder·빈 digest를 포함한
manifest는 `execution_ready=false` 상태로만 저장하고 실행 진입점에서 거부한다.

| 필드 묶음 | 필수 내용 |
|---|---|
| 식별·무결성 | `schema_version`, `experiment_id`, `run_id`, `question`, candidate ID, E/A/S, G/H/C/O, T1/T2/T3, manifest digest, 생성 시점, 이전 run과 관계 |
| 각 엔진 | 공개 source URL/commit, dirty 변경 여부와 의도한 patch 식별, binary digest, build mode·compiler·ISA, model/encoding ID, weight digest, weight 권리/출처, evaluator/search/backend/precision ID, 요청 옵션과 실제 적용된 전체 옵션 |
| 하드웨어 | CPU/GPU 모델과 개수, physical core/thread/affinity, RAM/VRAM 상한, OS/driver/runtime, GPU power policy, 장비 배정·교차 정책, 동시 상주/offload 정책, helper·tablebase 비용 한도 |
| 입력 | opening artifact digest·권리, 선택 방식·seed, 초기 상태+합법 이동열, 알려진 이력 시작점·반복/history-fill 정책, opening/pair/game ID, split ID |
| 프로토콜 | runner source/version/binary digest, UCI adapter version, 규칙 참조 ID/version, draw claim/automatic draw/terminal 우선순위, adjudication·tablebase·최대 수·process 장애 정책 |
| 시간·수명 | clock source·단위, 기본 시계·증분 또는 movetime, 시간 공제 경계, UCI overhead와 timeout grace, loading/compile/warm-up, request deadline·cancel/drain·newgame reset 정책 |
| 실험 계획 | 엔진/색 배정·실행 순서, 난수 seed 배정, tuning/holdout 구분, 최대 pair/game/wall-time/출력, retry 한도·무효 기준, 모든 worker와 자식 종료 조건 |
| 통계 | score 관점, logistic/normalized Elo 척도, pair/opening 군집 처리, fixed-sample 또는 SPRT/GSPRT 구현·버전, H0/H1·alpha/beta 또는 CI 방법, 표본·예산·중단·승격 기준 |
| 증거 | PGN/event/metrics/summary artifact digest, 논리 저장 경로, 시작/종료 시각, 실사용 자원·실패 수·중단 원인, 해석 가능 여부와 제외 사유 |

실제 경로는 저장소 밖 생성물 루트에서 해석한다. 공유 manifest에는 개인 절대 경로,
계정·호스트·비밀을 넣지 않는다. 가중치·책·로그의 digest 확인은 해당 파일 접근이
명시적으로 허용된 실행에서만 수행한다. 비밀 파일은 manifest 수집 대상이 아니다.

실행 전 잠근 입력 manifest와 실행 후 영수증을 분리한다. 실행 중 산출물 경로·실패·
관측값을 추가해도 엔진·opening·시계·표본·검정의 잠금 값은 바꾸지 않는다. 영향을
주는 입력이 바뀌면 새 manifest digest와 새 실험군을 만든다. 중단 run의 재개는 같은
입력·완료 pair·실패 정책 확인 후 수행하며 resume 사건과 실제 새 process를 기록한다.

## 3. R0-A: LC0 기준선 어댑터

**입력:** 잠긴 LC0 identity, 호환 가중치·backend, 자원 상한, UCI·오류 정책.
**출력:** 엔진 시작 영수증, 적용 옵션 스냅샷, bounded smoke 기록, baseline artifact ID.

1. 빌드 출처와 binary/weight digest를 확인한다. 엔진 표시 이름만으로 버전을 식별하지 않는다.
2. UCI handshake와 준비 완료, `ucinewgame`, 상태 전달, `go`, `stop`, `bestmove`,
   `quit`의 성공·실패·timeout 경로를 구현한다. 대기 단계마다 deadline을 둔다.
   기본 명령 의미는 [UCI 프로토콜 참조](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html)와
   선택한 runner의 실제 동작을 대조한다. Stockfish 전용 옵션을 LC0 옵션으로 가정하지 않는다.
3. 요청 옵션과 실제 적용 옵션을 대조한다. backend가 의도한 모델을 지원하는지,
   실제 선택된 장치·정밀도가 무엇인지 확인한다. CPU fallback이나 다른 가중치 사용은
   성공으로 숨기지 않는다.
4. stdout의 프로토콜과 진단 기록을 분리한다. malformed 출력, 잘못된 promotion,
   duplicate `bestmove`, 종료 뒤 출력, process 비정상 종료를 유형화한다.
5. 시작·중단·재시작을 검증하고 process tree가 남지 않는지 확인한다. 재시작이
   결과를 바꾸거나 상태를 잃으면 새 사건·game attempt로 기록한다.

**완료 근거:** 고정 identity로 정상·실패 경로가 재현되고 준비·착수·종료 상태가
감사 가능하다. 해당 플랫폼에서 실제로 실행한 범위만 통과로 표시한다.
**중단:** identity 불일치, 미지원 backend, hidden fallback, 자원 초과 또는 종료 불능.
R0에서 LC0 내부 모델·탐색 최적화는 변경하지 않는다.

## 4. R0-B: 완전 시작 상태와 pair 계획

**입력:** 권리·출처가 확인된 opening 묶음, 완전 상태 계약, seed와 실행 순서 정책.
**출력:** immutable opening manifest, pair/game 계획, 독립 상태 복원 감사 결과.

한 opening은 초기 규칙 상태와 거기서의 합법 이동열로 복원한다. 보드·실제 차례·
캐슬링·앙파상·halfmove/fullmove 카운터와 알려진 반복·모델 입력 이력을 보존한다.
임의 중간 FEN은 그 이전의 이력을 알려 주지 않는다. 중간 FEN을 허용한다면
`history_origin`과 알려진 범위를 명시하고 그 범위 밖 반복을 추정하지 않는다.
모델의 history-fill과 규칙 판정용 이력을 별도로 선언한다.

각 pair는 같은 완전 P를 두 번 사용한다. `game_a`는 후보 백/LC0 흑,
`game_b`는 LC0 백/후보 흑으로 엔진 배정만 교환한다. P의 실제 차례나 기물 색·
보드를 변환하지 않는다. 두 엔진에 동일 복원 이동열을 전달하며 runner가 정확한
이동열을 전달하는지 출력 사건과 PGN으로 대조한다.

pair의 game 순서·엔진 장비 배정은 사전 계획으로 균형 있게 교차한다. 확률적
엔진의 seed는 게임·색·실행 순서와 함께 기록한다. 같은 결정적 상태의 단순 반복을
독립 opening으로 세지 않는다. 새로운 결과를 본 뒤 유리한 opening만 고르지 않는다.

**완료 근거:** 계획의 두 판 상태 digest가 같고 엔진/색 배정만 다르며,
독립 규칙 참조의 복원·합법 수와 PGN이 일치한다. 감사기는 비교하는 엔진과 같은
상태 전이 함수만 호출해서 독립성을 주장하지 않는다.
**중단:** 미상 이력의 추정, 특수 수·카운터 손실, 다른 P 사용, 잘못된 색 교대.

## 5. R0-C: 시계·자원·취소 경계

**입력:** 잠긴 T1/T2/T3, 자원·warm-up·drain 정책과 엔진 어댑터.
**출력:** 수당 시계 사건, 자원 사용 기록, 잔여 작업 감사와 fairness 판정.

runner의 monotonic clock를 기준으로 `go`를 전달한 시점부터 유효한 `bestmove`를
수신한 시점까지 기록한다. 실제 시계 공제에 사용하는 쓰기/수신 경계·pipe overhead·
grace는 runner별로 확인하고 잠근다. `go`→`bestmove` 측정과 엔진 내부 자체 시계는
서로 대체하지 않는다. T1의 동일한 것은 기본 시계와 증분의 허용값이며, T2의
동일한 것은 매 수 movetime의 허용값이다.

해당 구간에는 특징 생성, cache lookup/restore/rebuild, helper, queue wait,
H2D/D2H, GPU 완료, selection/backup과 출력 비용이 모두 들어간다. `position`
처리의 정확한 상태 복원 비용은 따로 기록하고 필요하면 상태 전달→출력까지 측정한다.
`position` 단계의 평가·추측 실행으로 계산 비용을 시계 밖에 숨기지 않는다.

loading·compile·일반 warm-up은 양쪽 동일 정책을 적용한다. warm-up 입력은 평가
opening과 분리하고 생성 원칙·횟수·시간·메모리와 이후 cache reset을 기록한다.
평가 포지션·후속 수를 사전 분석하지 않는다. Ponder와 상대 차례의 helper·신경망
평가·speculation은 끈다. 온라인 가중치 변경과 게임 간 결과·상대 학습 유입을 금지한다.
게임 안 history/correction은 manifest에 켠 경우 자기 시간에만 갱신하고
`ucinewgame`에서 초기화한다. raw eval cache와 검색 correction을 구별한다.

요청은 generation·deadline·모델·루트 수명을 갖는다. `stop`, 착수, timeout,
`ucinewgame`, process 종료의 각각에 대해 pending·in-flight·완료-but-unconsumed
요청을 취소/폐기하고 예약을 한 번만 해제한다. 이전 generation의 결과는 다음 루트에
backup하거나 캐시를 통해 다음 생각 시간의 숨은 이득으로 사용하지 않는다.

취소 불가 GPU 구간은 launch부터 실제 완료까지 기록한다. 후보 구현은 가능한 한
새 작업 중단·drain을 자기 시간 안에 완료한 뒤 `bestmove`를 낸다. `bestmove` 뒤
잔여 계산이 남는 backend는 폐기 결과의 격리만으로 자원 공정성이 증명되지 않는다.
상대 차례의 GPU 점유·대기와 다음 수 유입을 감사하고, 사전 잠근 runner 정책에서
공정한 중단·배정이 검증되지 않으면 `fairness_unverified`로 두어 강도 해석을 보류한다.
드레인 대기·모델 교체 비용을 한쪽에만 시계 밖으로 빼지 않는다.

단일 GPU의 기본 계획은 한 번에 한 대국이다. 생각 중인 엔진 외 process가 GPU
평가를 실행하지 않도록 검증한다. 동시 상주는 RAM/VRAM·메모리 경쟁·offload를
확인하고, 들어가지 않으면 동일 장치 별도 제공·배정 교차 또는 재정의한 실험군을
잠근다. 두 엔진의 같은 `Hash` 숫자를 총 메모리 동일의 증거로 쓰지 않는다.

**완료 근거:** 단위·시계 경계·실제 적용 상한·자식 종료를 확인하고,
취소/중복/늦은 완료/newgame reset의 독립 사건 대조가 통과한다.
**중단:** 자기 시간 밖 평가, 자원 경쟁의 숨김, 이전 요청의 새 루트 반영,
시간초과 은폐, 증분/기본 시계 오해 또는 측정하지 않은 async 잔여 작업.

## 6. R0-D: 종료·실패·pair 회계

**입력:** 독립 규칙 계약, 사전 draw/adjudication/장애·재시도 정책.
**출력:** 모든 game attempt의 종료 상태, 원시 PGN과 pair ledger.

규칙 terminal은 Rules가 소유한다. runner는 독립 참조로 수·결과를 감사한다.
체크메이트, 스테일메이트, dead position, claim 가능한 반복/50수와 자동 반복/75수의
구별·우선순위를 선택한 규칙 계약과 대조한다. claim을 자동 적용하는 runner 설정은
명시한 평가 정책이지 모든 claim이 자동 종료된다는 뜻이 아니다. tablebase 판정과
평가점 기반 adjudication도 규칙 terminal과 별도 종료 원인으로 기록한다.
초기 평가점 adjudication 비활성은 권고이며 실제 실행값은 manifest로 잠근다.
FIDE 규칙에서 3회 반복/50수 claim과 자동 5회 반복/75수는 서로 다르며, 75수에서
마지막 수의 체크메이트 우선순위도 감사한다. 반복 판정의 EP 차이는 실제 합법 EP
가능성 등 선택한 규칙의 동등성 기준으로 처리한다. 단순 material 휴리스틱을
완전한 dead-position 판정으로 주장하지 않는다.
참조 기준은 [FIDE Laws of Chess](https://handbook.fide.com/chapter/e012023)이며
실제 runner의 claim 적용 프로필·해당 규칙 버전을 실행 전에 기록한다.

| game 상태 | 기록과 score 처리 |
|---|---|
| `rules_terminal` | 정확한 규칙 종료 이유·최종 상태·수순과 W/D/L. 결과를 1/0.5/0으로 계산한다. |
| `protocol_adjudicated` | 사전 잠근 runner claim/tablebase/기타 판정의 정책 ID·근거. 규칙 자체 종료와 구별한다. |
| `engine_loss` | 불법 수·crash·timeout 등 원인, 책임 엔진과 수당 시계. 사전 정책대로 loss에 포함하며 빈도를 별도로 보고한다. |
| `infrastructure_invalid` | 양 엔진 외 장비/runner/storage 장애의 객관 증거. 원시 attempt를 보존하고 pair 전체를 새 attempt ID로 재실행한다. |
| `incomplete` | 수·시간·실행 예산 또는 사용자 취소 등으로 미완료. 자동 0.5를 넣지 않으며 완료 pair 통계에서 제외한 범위를 표시한다. |
| `contract_invalid` | 상태·모델·관점·시계·자원/manifest 불일치. 해당 결과의 강도 해석을 중단하고 범위를 명시한다. |

게임 하나만 선택적으로 재시작하지 않는다. 무효 pair의 두 판 attempt와 새 pair
attempt를 모두 보존하고 실제 완료 pair만 최종 pentanomial에 한 번 들어간다.
엔진 패배를 인프라 장애로 돌려 다시 돌리지 않는다. 양쪽 장애·동시 timeout·
최대 수 도달의 정책도 첫 결과 전에 정의한다. 정의되지 않은 상태는 정상·무승부로
변환하지 않고 incomplete/contract 오류로 반환한다.

**완료 근거:** WDL·종료 이유·attempt 수와 pair ledger가 일치하고, 잘못된 한 판
삭제·중복 계산·incomplete→draw 전환을 감사기가 검출한다.
**중단:** 유리한 판만 보존, 책임 없는 crash 무효화, 결과/수순 모순 또는 장애 증거 손실.

## 7. R0-E: pair 통계와 사전 계획

후보 관점 `s=(W+0.5D)/(W+D+L)`과 pair 합계 0/0.5/1/1.5/2의 빈도
`n0..n4`를 모두 보존한다. `M=sum(n0..n4)`, 완료 게임 수 `2M`,
`s=(0n0+0.5n1+1n2+1.5n3+2n4)/(2M)`을 원시 WDL과 교차 확인한다.
1점 pair 안의 두 무승부와 승/패가 구별되므로 WDL을 지우지 않는다.

일반 logistic Elo는 `400*log10(s/(1-s))`이며 fixed 상대·장비·시계·opening
분포에 대한 차이다. `s=0/1`의 유한 점추정치를 임의 생성하지 않는다. normalized
Elo와 같은 이름·경계로 혼용하지 않는다. pair 상관과 같은 opening의 반복 사용에
따른 추가 군집 상관을 반영하는 방법·구현을 고정하고 작은 손계산 fixture 및
독립 통계 참조와 대조한다. 조건이 다른 실험군은 합산하지 않는다.

개발 순차 검정은 구현·척도·H0/H1·alpha/beta·최대 pair·중단 규칙을 결과 관측
전에 잠근다. 지원하지 않는 paired SPRT/GSPRT를 자체 이름만 붙여 구현했다고
기록하지 않는다. 최종 holdout은 튜닝 opening과 분리한 fixed sample을 우선
권고한다. 최소 검출 효과·pilot 분산·예산으로 표본을 결정하고 holdout을 반복
선택·튜닝에 사용하지 않는다. 순차 중단 뒤 단순 fixed-sample CI로 선택 효과를
무시하지 않는다.

핸드오프의 100쌍 smoke, 2,000쌍 final, T1 60+0.6/180+1.8초,
T2 100/1,000ms, 95% CI 하한 > 0의 승격 조건은 **사전 선택할 제안**이다.
이 문서 작성으로 실행값·허용 시간·자동 승격 기준이 확정된 것은 아니다.

**완료 근거:** fixture의 pair/WDL/score 일치, 통계 참조 대조, 고정 표본·순차·
불확정 상태의 서로 다른 보고가 가능하다.
**중단:** 결과를 보고 표본·척도·opening·경계를 바꿈, 무효/미완료를 무승부로
추가함, 표본 의존성을 무시함 또는 holdout 누출.

## 8. R1: 전체 경로 계측

**입력:** 자체 C02/C03의 형식·독립 수치·정확성 인수를 충족한 모델·가중치·search,
잠긴 manifest·시계·자원·취소 계약과 계측 예산. 자체 D02 계측은 외부 LC0의
adapter·paired 대국·통계 전체 완료를 선행 조건으로 삼지 않는다. 외부 LC0 자체를
계측하는 run은 해당 LC0 identity·실행에 필요한 R0 계약을 별도로 충족한다.
**출력:** 요청 사건·pipeline latency/throughput·실제 batch·메모리·실사용 평가 보고.
계측 자체가 의미 있는 속도/선택 변화를 만들면 sampling·overhead와 대조 run을 기록한다.

각 요청은 request/generation/model/input/precision/budget ID로 연계한다.
CPU 준비 시작/끝, queue 진입/배정, H2D 시작/완료, GPU launch/실제 완료,
D2H 완료, result 검증, cache 저장, search 소비/backup, 취소·폐기 시각을
가능한 계측 수단으로 기록한다. 측정 불가능한 구간은 추정으로 표시하고 합산해
관측값으로 만들지 않는다. async host launch 지연은 GPU 완료 지연이 아니다.

| 계측 묶음 | 계산·해석 계약 |
|---|---|
| 시간 | 수당 wall-time·request P50/P95/P99·queue wait·critical-path wait·정상 상태 throughput. 겹치는 단계 합을 전체 시간/처리량으로 사용하지 않는다. |
| 모델 | parameter/active parameter·token/slot/iteration·precision·shape, MAC/FLOP 정의. 반복 횟수/낮은 정밀도 차이를 숨기지 않는다. |
| 자원 | CPU/GPU idle·active, peak/resident RAM/VRAM, 전송 바이트, batch 분포, eviction·복원 비용·cache age. GPU utilization만으로 유용성을 판단하지 않는다. |
| 실제 평가 | 물리 실행 횟수, 완료된 고유 평가 키 수, search에서 실제 소비한 고유 평가 수, 사용되지 않은 선행 평가·취소·중복 실행을 별도로 기록한다. |
| search | 실제 visit/backup·예약/in-flight·반영하지 않은 late 결과·유용한 요청이 선행 작업에 막힌 시간. eval hit나 예약을 새 NN 계산·완료 방문으로 세지 않는다. |
| 품질 | WDL calibration·policy 순위·교사 regret·유일 방어 실패·fresh/warm 편차·NaN/Inf/invalid 출력. 평균뿐 아니라 실패군과 꼬리를 보존한다. |

고유 평가 키는 [CONTRACTS](CONTRACTS.md)의 실제 입력·모델·정밀도·계산
예산·경로 계약을 따른다. 같은 입력을 중복 계산한 것은 물리 실행 비용에 남기고
고유 평가 수를 늘리지 않는다. cache hit는 cache 사용으로 남기며 새 계산 수에
포함하지 않는다. 근사 warm-state와 fresh는 같은 exact 키로 묶지 않는다.
하나의 raw 평가를 여러 edge가 사용한 횟수와 고유 소비 수를 구별한다.

cold start, 부모→자식, sibling 묶음, transposition, eviction 후 재계산,
특수 수·king/threat delta·긴 누적 trace를 나눠 측정한다. batch 1/4/16/64/256은
시작 후보이며 예산을 넘는 값은 생략 이유를 적는다. microbenchmark batch와 실제
search batch 분포를 둘 다 남긴다. trace replay의 결과는 실시간 탐색의 변화된
분포에 대한 증거가 아니므로 최종 same-time 대국을 별도로 실시한다.

**완료 근거:** 비용 경계·누락·계측 overhead가 명확하고, 소비된 평가/중복/취소와
실제 wall-time이 대조 가능하다. CPU/GPU 병목을 관측 근거로 설명할 수 있다.
**중단:** 물리 완료와 launch 혼동, eval hit로 처리량 부풀림, 중첩 시간 단순 합산,
상한 초과, 실제 소비를 식별할 수 없음 또는 정확성/시계 계약 위반.

## 9. 단계 인수와 보존

R0 인수는 강도 개선이나 production 준비 완료가 아니다. baseline identity,
pair 상태·배정, 시계·자원·취소, 종료·실패, 통계·artifact 무결성을 각각
실제 실행한 독립 대조의 범위로 기록한다. R1 인수는 정확한 측정 경계와 병목
보고이며 R2의 선택 근거가 된다. E/A/S 변경마다 별도 실패·수명 검사를 적용한다.

최종 보고는 입력 manifest와 artifact digest, 전체 PGN·event·attempt ledger,
WDL/pentanomial/score/Elo/CI·중단 원인, 수당 시간·실사용 자원·helper 비용,
규칙·계약 오류와 미검증 범위를 포함한다. 작은 공개 fixture·재현 설정·검토된 요약만
저장소에 두고 원시 PGN·로그·가중치·대형 보고서는 지정 생성물 루트에 보존한다.
로컬 검사·실제 CI 실행·재사용 검증·smoke·대국 강도는 서로 다른 증거로 보고한다.
