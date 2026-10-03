# 실험·대국 계약

이 문서는 [RoveZero v0.2.0 핸드오프](reference/RoveZero_Handoff_v0.2.0_KO.md)의 2장, 10~12장, 14~17장을 실험의 목표·연구 분류·해석 경계로 정리한다. 현재 사용자 결정과 실제 작업 의존성은 [IMPLEMENTATION-DIRECTIVES](IMPLEMENTATION-DIRECTIVES.md)가 우선한다. 상세 manifest·러너·계측·통계 구현은 [EVALUATION-PROTOCOL](EVALUATION-PROTOCOL.md), 데이터·미세조정·새 구조 실험은 [TRAINING-PLAN](TRAINING-PLAN.md), 전체 76개 후보의 최소 작업·검증·보류 조건은 [CANDIDATE-REGISTER](CANDIDATE-REGISTER.md)를 따른다.

구현·학습·프로파일링·대국을 수행했다는 보고가 아니다. 상태와 책임은 [ARCHITECTURE](ARCHITECTURE.md), 공통 계약은 [CONTRACTS](CONTRACTS.md), 결정 상태는 [DECISIONS](DECISIONS.md), 일반 변경 규약은 [ENGINEERING-STANDARDS](ENGINEERING-STANDARDS.md)를 따른다. 최소 공통 계약의 내용·revision·변경은 총괄이 직접 소유한다. TASK-A03은 그 계약을 적용·검증하는 작업이며 독립 계약 결정권자가 아니다.

A~F의 클라우드 개발에는 GPU가 없을 수 있다. CPU/mock·가능한 명시적 CPU 신경망
참조·fixture·계측 도구·학습 recipe를 먼저 개발하고 별도 GPU 검증 환경과 구분한다.
초기 RTX 4050 인수 이력을 보존하고 후속 GPU 벤치마크는
[외부 RunPod 준비 계획](research/RUNPOD-BENCHMARK-PLAN.md)에서 장비·예산을 잠근다.
사용자 후속 지시로 GPU A/B 전에 [의미 보존 CPU 최적화](research/PRE-RUNPOD-OPTIMIZATION.md)를
먼저 인수했다. 다음 GPU 실행에서 새 소스의 실제 수치·종단 검사를 확인한 뒤 baseline으로
잠근다. 현재 초기 총예산·Community/no network volume·외부 회수 조건은 준비 계획을 따른다.
GPU 검사의 skip·미실행을 통과로 처리하지 않는다. 실제 GPU·학습·정식 대국 인수는
총괄 I02가 지원 환경의 정확한 SHA·설정·자원·실행 증거로 확인한다. 단계별 인계는
구현 지시서의 클라우드 개발 규약을 따른다.

## 목표와 사전 잠금

최종 목표는 LC0와 동일한 전체 장비·시간 상한에서 더 높은 득점률을 얻는 것이다. 같은 시작 상태의 색 교대 두 판을 기본 비교 단위로 사용한다. FLOPs·NPS·퍼즐 정확도·압축률·kernel 시간은 원인 분석 자료이며 강도 승격을 대신하지 않는다.

실험 전에 질문, 변경 한 가지와 E/A/S 분류, G/H/C/O 경로, 고정 조건, 측정 경계, 실패 처리, 예산 상한·중단 규칙을 manifest에 기록한다. LC0 바이너리 commit·빌드, 네트워크 SHA-256, search/backend/precision/options, 후보 commit·가중치, 장비·driver·전력 정책, opening·seed·시계·러너·통계 버전을 잠근다. `LC0 최신`이라는 이름만으로 기준선을 식별하지 않는다.

독립 Rust workspace·자체 규칙 코어·자체 작성 코드의 MIT 정책·최소 PUCT 기준선은 사용자 결정이다. 외부 체스 라이브러리는 독립 대조·테스트용으로 사용하고 LC0 fork·코드 결합을 기본값으로 삼지 않는다. 자체 코드의 MIT와 외부 코드·가중치·데이터·교사·opening 권리는 각각 확인한다. mock PUCT 구현, 실제 Rust 신경망 추론, 외부 UCI LC0 연결과 실제 대국 증거를 구분한다.

첫 호환 가중치는 [WEIGHT-SELECTION](WEIGHT-SELECTION.md)의 CSSLab Maia1 v1.0 `maia-1900.pb.gz`로 선정한다. 원저자의 weights GPL 적용 명시와 직접 확인한 6×64 SE 구조를 근거로 호환·runtime 기준선에 사용하며, 인간 수 예측 모델의 선택을 최종 LC0 강도 가중치 선정으로 해석하지 않는다. T70 `703810`은 개별 권리 미확인으로 보류한다. 실제 GPU 호환 인수·재배포의 GPL source/notice 의무·F02 실행 조건은 각각 충족해야 한다. RTX 4050 Laptop 6GB 조건은 실행·학습·대국의 시간·CPU/RAM/VRAM 상한을 대신하지 않는다. 학습 도구의 언어, 실제 학습/GPU 예산·모델 크기·반복·cache precision·대국 시계·표본·검정·승격 수치는 실행 전에 잠근다. 문서의 권고 숫자를 설정이 잠긴 사실로 사용하지 않는다.

## 현재 구현과 내부 대조의 순서

총괄이 최소 공통 계약을 발행한 뒤 A·B·C01·D01·E·F01은 mock·fixture로 병렬 구현한다. 외부 LC0 기준선의 가중치·backend·대국 설정 잠금은 실제 외부 비교의 진입 조건이며, 독립 Rust 규칙·상태·mock 탐색·계약·큐·데이터 감사 구현 전체를 막는 조건이 아니다. 실제 신경망 비교는 TASK-C02의 단일 형식·권리·호환과 TASK-C03의 CPU reference·목표 GPU·Rust 연결을 검증한 뒤 시작한다.

| 내부 조합 | 가중치 | 탐색 | 비교 질문 |
|---|---|---|---|
| CONTROL-0 | 권리·형식·호환을 확인한 원본 고정 W0 | 자체 baseline PUCT S0 | 실제 RoveZero 내부 기준. mock 결과는 별도 G1 근거다. |
| CONTROL-1 | 동일 W0 | 새 탐색 S1 | CONTROL-0 대비 탐색 변경의 효과. weight·encoder·precision·runtime·자원·시계를 고정한다. |
| CONTROL-2 | TASK-F02의 미세조정 W1 | 동일 S1 | CONTROL-1 대비 학습 변경의 효과. architecture·입력·head·search·runtime 조건을 유지한다. |

TASK-B02의 baseline/variant가 CONTROL-0→1의 탐색 대조를 담당한다. TASK-F01은 학습 데이터 계약·누출 감사이며 baseline PUCT의 작업 ID가 아니다. TASK-F02는 원본 가중치의 실제 Rust 추론과 새 탐색 대조를 검증한 뒤 미세조정을 수행한다. TASK-F03의 첫 architecture는 TASK-F02 재현·검증 뒤 단일 변수를 선택한다. 반복 잠재 연구 CARD-C02와 가중치 호환 작업 TASK-C02, warm-start 연구 CARD-C03와 실제 NN 추론 작업 TASK-C03을 혼동하지 않는다.

TASK-D03의 runtime baseline/variant는 **동일 W·동일 S**에서 계측으로 고른 실행 개선 한 가지를 비교하는 독립 축이다. runtime·search·weight를 같은 A/B에서 함께 바꾸지 않는다. 평가 선택·순서가 바뀌면 S를 표시하고 수치·precision·근사가 바뀌면 A를 표시한다. 내부 비교 통과가 외부 LC0 대비 강도나 모델 승격을 뜻하지 않는다.

## 핸드오프 R0~R7 연구 분류

v0.2.0 17.1절의 R0~R7은 이전 M0~M6 및 [PlyZero v0.1.0](reference/PlyZero_Handoff_v0.1.0_KO.md)의 구조 우선순위를 개정한 연구 분류다. 현재 담당 작업의 착수 순서와 CONTROL-0→1→2→TASK-F03는 위 사용자 결정을 따른다. R0에서 외부 비교의 설정을 재현하고 R1에서 실제 실행 경로를 계측한 뒤 성능 해석을 한다. 행 연결은 연구를 이어갈 근거이며 현재 20개 TASK 전체의 실행 순서나 최종 승격이 아니다.

| 단계 / 경로 | 독립 변경 또는 산출물 | 고정 조건과 다음 단계 근거 |
|---|---|---|
| R0 / 외부 비교 공통 | LC0·pair runner·manifest 재현 | 외부 비교 모델·commit·backend·장비·opening·시계 고정; 코드 변경 없이 UCI·시간·pair/PGN 기록 검증. 독립 Rust·mock 개발의 전면 선행 조건은 아니다. |
| R1 / G | CPU 준비·GPU 큐·전송·kernel·backup 계측 | 기존 가중치 고정; 병목·실제 batch 분포·측정 경계 확인 |
| R2 / G | 실행·큐·node layout·선택 경로 중 한 변경 | 같은 모델과 필요한 검색 계약; 수치·수명·취소 계약 및 전체 지연 검증 |
| R3 / G | 관계 입력 또는 작은 잠재 구조 하나 | TASK-F02 뒤 TASK-F03에서 선택할 새 architecture 후보. 동일 데이터·학습 예산·검색 설정의 대응 square-token 기준; 품질/속도와 equal-time 대국 |
| R4 / H | 잔차 gate·시간 관리자·작은 helper 하나 | GPU 모델 고정; host 비용·calibration·시간초과·각 변경 효과 분리 |
| R5 / G·H | recurrent·warm-start | TASK-F03 이후 선택한 CARD-C02/C03 연구. fresh 기준부터 부모 state 전송·상주량·경로 편향·정확 cache 구분·deadline을 검증하고 후속 변수를 분리한다. |
| R6 / C | CPU 증분·PVS·lazy accumulator 등 비교 | CPU와 GPU 기준선의 해석 분리; 주 GPU 연구의 대체로 자동 승격하지 않음 |
| R7 / 선택 | 통과 요소 결합과 제거 실험 | 독립 holdout·같은 시간·색 교대·전체 실패 기록; 각 요소 ablation |

R2에서도 평가 선택·순서가 바뀌면 S를 표시한다. 이미 있는 외부 LC0 옵션을 꺼 둔 약한 기준선과 비교해 새로운 최적화라고 주장하지 않는다. 자체 호환 실행의 구현과 외부 LC0 runtime 변경은 별도 대상이다. 증분·양자화·동적 반복·helper를 baseline 없이 동시에 넣지 않는다. 다음 연구 단계로 가는 근거와 최종 대국 승격은 구분한다.

## 시작 상태와 시간 트랙

| 트랙 | 양쪽에 제공하는 시간 | 해석 |
|---|---|---|
| T1 / 주 평가 | 동일 기본 game clock + 동일 increment | 모델·검색·캐시·시간 관리가 결합된 실제 엔진 강도 |
| T2 / 통제 | 동일 포지션에서 매 착수 동일 movetime | 수당 시간 배분을 통제한 비교 |
| T3 / 진단 | 동일 고유 신경망 평가 수 또는 명시한 검색 예산 | 평가 품질·실행 속도 원인 분석; 최종 Elo 대체 불가 |

T1은 허용 시간이 같다는 뜻이며 실제 소비 시간이 같다는 뜻은 아니다. 서로 다른 트랙·시간 제어·장비의 결과를 합쳐 하나의 Elo로 만들지 않는다.

하나의 시작 포지션 P에서 외부 비교 1판은 RoveZero 백/LC0 흑, 2판은 LC0 백/RoveZero 흑으로 실행한다. 내부 CONTROL 대조도 두 구성의 엔진 배정만 바꾸는 같은 규칙을 사용한다. 두 판의 보드·실제 차례·캐슬링·앙파상·카운터·제공 이력은 같다. 보드 반전이나 기물 색 변환으로 만들지 않는다. `pair_id`와 `opening_id`를 보존하고 실행 순서를 균형 있게 바꿔 발열·순서 영향을 줄인다.

동일 PGN 경로를 두 엔진에 전달한다. FEN만 쓰면 이전 반복 이력이 없다는 한계와 초기화 정책을 명시한다. opening 선택 기준·seed를 사전 고정하고 다양한 시작 상태를 사용한다. 같은 결정적 설정의 단순 반복을 새로운 독립 정보로 취급하지 않는다. 기존 러너를 우선 검토하고 실제 PGN으로 색 교대를 감사한다.

## 시간과 자원 공정성

`go` 이후 feature 생성, cache 조회·복원·재구성, CPU/GPU 전송, queue 대기, GPU 완료, 검색·backup, 결과 출력까지 wall-clock으로 잰다. 비동기 launch 시간을 완료 시간처럼 보고하지 않는다. 단계별 시간 합이 중첩된 pipeline의 전체 처리량과 같다고 가정하지 않는다.

- 가중치 loading·compile·일반 warm-up은 양쪽 동일 정책으로 대국 전에 허용하고 별도 기록한다. 평가 포지션 사전 분석, `position` 처리의 숨은 평가와 상대 차례의 helper/speculation은 허용하지 않는다.
- Ponder는 양쪽 끈다. 자기 생각 시간의 선행·형제 평가와 게임 내 cache/tree 재사용은 허용하지만 모든 비용을 자기 시계에 넣는다. 새 게임의 게임별 cache·학습 상태는 초기화한다.
- 고정 신경망의 온라인 가중치 변경, 다른 게임의 결과 유입과 영구 상대 학습은 주 평가에서 허용하지 않는다. 별도 프로토콜이 필요하다.
- 게임 안 search history/correction 갱신은 명시 옵션으로 허용한다. 자기 생각 시간만 사용하고 `ucinewgame`에서 reset하며 raw eval cache를 오염시키지 않는다. 온라인 가중치 변경과 구별한다.
- late 요청은 generation/deadline에 따라 버리며 착수 이후 잔여 계산의 중단·격리를 검증한다. 취소 불가 GPU 구간과 잔여 실행도 계측한다.
- 동일 CPU/GPU 모델·power/driver·physical-core·RAM/VRAM 상한을 적용한다. 같은 Hash 숫자가 같은 총 메모리라는 가정은 금지한다. H의 helper도 동일 상한 안에 포함한다.
- 양쪽 batch/thread/precision tuning은 최종 평가 밖 validation에서 허용한다. 단일 GPU는 한 번에 한 대국과 생각 중인 엔진의 GPU 계산을 우선한다. 함께 상주하면 메모리 경쟁·자동 offload를 확인한다.
- 양 모델이 함께 들어가지 않으면 동일 GPU 개별 제공과 장비 배정 교차 또는 재정의한 자원 조건을 기록한다. 한쪽 loading·transfer 비용만 시계 밖으로 숨기지 않는다.
- tablebase는 양쪽 같은 접근·probe·판정 정책 또는 양쪽 비활성으로 맞춘다. helper·tablebase 디스크 I/O·외부 composite 호출 비용도 포함한다.

## 통계와 실패 처리

RoveZero 관점 W/D/L을 보존하고 득점률 `s=(W+0.5D)/(W+D+L)`을 사용한다. 일반 logistic 상대 Elo는 `400*log10(s/(1-s))`이며 고정한 상대·장비·시간·opening 분포에 대한 차이다. 인간 절대 Elo로 환산하지 않는다. s=0/1이면 유한 점추정치를 만들어 넣지 않고 경계와 불확실성을 기록한다.

pair 합계 점수 0/0.5/1/1.5/2의 빈도 `n0..n4`, 총 pair 수, 원시 WDL을 보존하고 서로 교차 확인한다. pair 상관과 반복 사용 opening의 추가 군집 상관을 반영한다. 일반 logistic Elo와 normalized Elo는 다른 척도이므로 검정·결과에 명시한다.

순차 검정을 사용하면 검증된 paired 구현을 선택하고 H0/H1, alpha/beta, Elo 척도, 최대 pair·중단 규칙을 첫 결과 전에 잠근다. 결과를 본 뒤 경계·척도·표본 상한을 유리하게 바꾸지 않는다. 순차 중단 표본에 단순 고정 표본 CI를 붙여 선택 효과를 무시하지 않는다.

최종 확인은 튜닝에 쓰지 않은 holdout의 고정 표본을 우선 권고한다. pilot 분산과 최소 검출 효과로 표본·예산을 먼저 결정한다. 예산 끝에 결론이 없으면 불확정으로 보고한다. holdout을 반복 튜닝에 재사용하거나 유리한 순간에 고정 표본을 종료하지 않는다.

| 핸드오프의 제안 | 현재 상태 |
|---|---|
| T1 60초+0.6초 / 확인 180초+1.8초; T2 100/1,000ms | 장비별 overhead·timeout pilot 뒤 확정할 시작 후보 |
| smoke 100쌍=200판 | 동작·설정 확인 제안; 작은 Elo 개선 증거 아님 |
| final 2,000쌍=4,000판 | pilot·최소 효과·예산으로 바꿀 수 있는 계획 후보 |
| 주 T1 상대 Elo 95% CI 하한 > 0 + 규칙·시간·자원 조건 충족 | 승격 정책 후보; 실제 프로토콜에 사전 확정하기 전 자동 판정 규칙 아님 |

엔진 자체 불법 수·crash·timeout은 사전 정책대로 loss에 포함하고 빈도도 보고한다. 인프라 장애는 사전 정의한 객관 기준으로만 무효화하며 pair 전체를 새 run ID로 재실행한다. 불리한 판만 제거하지 않는다. 평가점 기반 조기 판정 비활성은 초기 권고이며, 수 제한 미완료 처리와 반복·50수 규칙의 러너 동작은 사전 검증·고정한다.

## 계측·훈련·결과물

cold start, parent→child, sibling 묶음, transposition, eviction 후 계산, 긴 갱신 trace를 구별한다. trace replay는 진단이며 변경된 탐색 분포의 실시간 대국을 대신하지 않는다. batch 1/4/16/64/256은 microbenchmark 시작 후보이며 장비 한도·생략 이유와 실제 search batch 분포를 별도 기록한다.

계측에는 precision·parameter/token/slot/iteration·FLOP 관례, P50/P95/P99 latency·queue·H2D/D2H·launch/sync, CPU/GPU idle·상주 메모리, cache hit/miss/restore/eviction/age, 고유 평가·실제 소비 평가·미사용/취소/중복, critical-path wait, WDL calibration·유일 방어 실패·fresh/warm 편차를 포함한다. cache hit를 새 신경망 계산으로 세지 않는다.

첫 실제 엔진 기준선은 권리·형식·출력 호환을 확인한 기존 W0와 자체 PUCT S0의 CONTROL-0이다. TASK-F02의 미세조정은 원본 architecture·입력·head를 유지하고 같은 새 search S1에서 비교한다. 소형 square-token 기준선은 TASK-F03의 새 architecture 비교군이며 기존 W0를 읽는 첫 runtime 기준선을 대체하지 않는다. 새 구조는 같은 데이터·출력 계약·학습 예산에서 대응 기준과 비교한다. 모델과 스케줄 변경을 분리하고 input/body/head/loss/optimizer/data를 한 번에 바꾸지 않는다. 자체 소형 모델 또는 내부 CONTROL 개선을 LC0를 이긴 결과로 보고하지 않는다.

game/opening 단위 train/validation/holdout을 먼저 나누고 인접 상태·중복·transposition 누출을 감사한다. teacher ID·version·budget, 완전 상태 또는 복원 이동열, label viewpoint·종료 이유·후보 policy/visits/WDL/Q를 기록한다. 대칭은 폰 방향·castling·EP의 규칙 의미를 보존해야 한다. 학습·교사 분석·재분석·자가대국·저장·튜닝 비용과 seed 선택 한계를 분리한다.

교사 엔진의 offline 데이터 생성과 실시간 외부 엔진 호출은 별개다. 후자는 composite/hybrid로 명명하고 모든 시간·자원을 기록한다. 코드·weight·data·opening의 출처와 권리를 각각 관리한다.

결과에는 전체 PGN, pair/opening/run ID, WDL·pentanomial·score/Elo/CI, 실제 수당 시간, crash/timeout/invalid/incomplete 기록, 파일 hashes·장비·driver·seed·통계 방법·중단 이유, track·helper 호출/시간·raw/corrected value·cache provenance·CPU/GPU 사용량을 남긴다. 작은 재현 설정은 소스에 둘 수 있으나 대형 데이터·로그·보고서·모델 산출물은 별도 위치에 보존한다.

## 해석 중단과 후보 보존

설명되지 않는 E 불일치, 잘못된 cache 공유, 자기 시간 밖 계산, WDL 관점 오류, 취소 요청 중복 방문, 잘못된 baseline backend가 발견되면 해당 Elo 해석을 중단하고 원인을 기록한다. 평균 오차가 작아도 드문 필패 전술이 늘면 보류한다. kernel만 빠르거나 선행 작업이 유용한 요청을 밀어내면 결합하지 않는다.

원문 A01~F30의 기존 76개 후보는 [등록부](CANDIDATE-REGISTER.md)의 CARD-A01~CARD-F30으로 보존한다. 실제 담당 TASK와 후보 CARD의 ID·소유권을 구분하고, 현재 20개 TASK의 범위에 76개 후보를 모두 자동 채택하지 않는다. 출처와 채택/실험/보류/기각 이유, 부정적·불확정 결과를 유지한다. CPU·GPU 직접·GPU 시스템 등급은 모두 원문의 미측정 추론이며 낮은 등급도 삭제하지 않는다. 76개를 독립 효과로 합산하지 않는다. 같은 net의 속도 개선, 같은 weight의 새 search, 새 weight·architecture의 강도, 시간 관리 효과를 각각 설명한다. 외부 저자 보고·적합성 추론·로컬 확인·실제 대국 결과를 구별하고 미실행 항목은 미실행으로 표시한다.
