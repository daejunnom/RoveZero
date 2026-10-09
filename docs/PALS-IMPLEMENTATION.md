# PALS 독자 엔진 구현과 인수 기록

사용자가 제공한 `RoveZero_PALS_Internal_Algorithms_KO.md`와
`RoveZero_PALS_Architecture_Flows_KO.md`는 설계 근거다. 문서 속 학습·실행 권고를
실제 실행 승인으로 확대하지 않는다. 이번 목표는 **실제 학습을 제외한 구현**이며,
기반 언어는 Rust다. 기존 Rules·PUCT·LC0 모델 경로와 과거 V1/V2 기록은 보존한다.

## 실행 경계

UCI 아래에서 PUCT, PALS, 자체 CPU 탐색을 선택한다. PALS는 P의 수순 제안,
C의 이탈·반박, 자체 CPU의 조건부 평가, P의 수선과 관련 결론 갱신을 수행한다.
합법 수·정확한 이력·종료 판정은 기존 Rules가 소유한다. PALS 작업 수나 CPU 노드를
PUCT simulation·visit 수로 표시하지 않는다.

학습 설계의 P/C/V와 대국 제품의 P/C를 구분한다. 제품 export와 실행에는 V를
포함하지 않는다. 자체 CPU_T와 CPU_R는 하나의 Rust PVS 코어를 공유하면서
프로필·평가 버전·노드·깊이·시간·TT 예산을 별도로 식별한다. 외부 UCI 엔진은
상대 및 독립 비교 경로다. 초기 학습 설계에는 외부 교사·기존 가중치가 포함되지 않는다.

정확한 CPU 결과·관측 기록과 GPU 표현 캐시를 분리한다. GPU 캐시를 회수해도 완료된
CPU 문제 결과는 남는다. 취소·세대 변경 이후 늦은 결과는 현재 루트에 적용하지 않는다.
물리 완료가 불명확하거나 격리된 실행은 입력·출력·workspace를 보존하며 재사용하지 않는다.

### 첫 value resolver와 교체 범위

첫 정책은 `pals-cpu-raw-restricted/0.1`이다. 등록된 CPU value namespace의 수용된
raw scalar와 실제 Rules terminal을 사용하고, 조사한 자식의 차례 관점 값을 부호 전환해
max를 계산한다. 수용된 frontier·부분 완료 값도 해당 범위와 함께 사용할 수 있으므로
요청 깊이 완료와 같은 뜻으로 기록하지 않는다. 미관측 값은 `None`이며 0·무승부로 채우지
않는다. CP/WDL 보정이나 P/C 값과의 평균은 하지 않는다. 실제 Rules 승리 착수를 우선하고,
같은 값에서는 terminal, 나머지 동점에서는 Rules 순서를 유지한다. 제한 후보의 추정값은
전체 게임의 bound 또는 완결 증명이 아니다.

`PalsResult`와 collector 원시 결과에 버전을 기록하고, UCI process 작업 영수증에는 버전과
의미 정의의 SHA-256을 추가한다. 시작 identity와 반환 결과의 정책이 다르면 거부한다.
과거 영수증의 필드 부재는 그대로 읽으며 해당 정책을 관측한 것으로 자동 채우지 않는다.
이 식별 추가는 현재 착수·점수 정책을 바꾸지 않는다. 후속 resolver 변경은 S 실험이다.

외부 UCI 상대 연결과 PALS 내부의 외부 `CPU_R` checker는 별도 접점이다. 첫 제품은 자체
`CpuSearcher`를 사용하며, 외부 checker의 capability·관점·raw mate/bound·단일 활성 go·
미지원 resume은 별도 인수한다. 현재 host whole-input K/V page의 실제 backing 소유권은
GPU resident page·record별 증분 재사용 또는 private latent warm-start의 인수를 대신하지 않는다.

## PALS V3 명세·lock·receipt

`rz-experiments::PalsRunManifestV3`는 `rz-pals-execution-v3/1` 도메인에서 JSON을
검증한다. `PalsInputLockV3`는 도메인과 canonical SHA-256을 고정한다. 기존 V1/V2의
codec·canonicalization·digest를 자동 변환하거나 PALS 실행으로 다시 해석하지 않는다.
PALS 계약 식별은 `pals/0.1`이며 기존 policy/WDL 평가 계약 revision `0.1`과 별개다.

엔진 endpoint는 PALS, 자체 CPU, 외부 UCI 비교 대상으로 나눈다. PALS 모델은 입력·head·
구조·구현·정밀도·backend·batch·고정 epoch와 가중치 출처를 기록한다. 결정적 mock,
미학습 초기 파라미터, 기존 학습 체크포인트는 서로 다른 타입이다. 파일 메타데이터 검증과
실제 파일의 hash 확인·모델 로딩·provider readiness는 다른 증거다.

생성 PGN의 native 참조는 실행 설명을 보존한다. Core의 공통 `ArtifactRef`에 연결할
때에는 `source`만 잠긴 Fastchess producer의 공개 HTTPS 소스 URL로 투영하고,
path·SHA-256·bytes·license는 유지한다. `pgn_provenance`에 원래 native 참조와
producer 소스 commit·바이너리 SHA를 따로 남긴다. URL은 로컬 PGN의 공개 다운로드
위치를 뜻하지 않으며, 공통 URL 검증·실패·시계·물리 수명·인수 조건을 완화하지 않는다.

| 비교 질문 | 고정·변경 조건 |
|---|---|
| `System` | PALS 전체 구성과 자체 CPU 또는 BT4/LC0 구성의 효과. 모델·탐색 전체가 달라지는 비교를 단일 축 효과로 보고하지 않는다. |
| `InternalModel` | PALS 모델 구성만 변경. 자체 CPU·탐색·runtime·자원은 동일하다. |
| `InternalSearch` | PALS 탐색만 변경. 모델·자체 CPU·runtime·자원은 동일하다. |
| `Runtime` | PALS runtime·pool 구성만 변경. 모델·탐색·자체 CPU·자원은 동일하다. |

명세가 선언한 변경 축과 두 endpoint의 실제 구성 차이가 일치해야 한다. 소스 SHA와
바이너리 hash는 실행별로 기록하되, 그것만으로 구현 의미의 동등성을 주장하지 않는다.
바이너리 내용이 바뀌었는데 어떤 구성의 구현 식별도 바뀌지 않은 경우는 `Endpoint`
변경으로 남겨 단일 축 비교를 거부한다. 같은 바이너리의 저장 경로 차이는 변경 축으로
세지 않는다. 이러한 선언 대조는 실제 소스 diff와 빌드·파일 hash 인수를 대신하지 않는다.
PALS 도입은 모델·탐색 실험이므로 5% 의미 보존 회귀 문턱을 도입 통과 조건으로 사용하지 않는다.

명세와 결과 영수증은 이번 범위에서 `training_executed=false`를 요구한다. 기존 학습 자산을
참조할 수 있는 schema와 실제 이번 실행에서 학습을 수행한 기록을 구분한다. CPU/mock CI
성공은 실신경망 수치 검사·GPU 실행·학습 성공으로 바꾸지 않는다.

## 유한 실행과 관측

첫 대국 명세는 표준 시작 상태, 흑백 교환 두 판, 120초+1초 피셔, 최대 256 ply,
동시 대국 1개, 게임마다 프로세스 재시작, 전체 최대 15분과 정리 최대 30초다.
Ponder·평가값에 의한 조기 판정·두 판으로부터의 Elo 주장을 허용하지 않는다.
CPU affinity·thread 수, memory.high/max, swap, 요청 GPU와 device 할당 상한을 선언한다.
상태·수순 chunk·상황·관측·작업·역할 상태·메모리 page·큐의 최대 개수도 선언한다.

`PalsRunReceiptV3`는 요청 옵션, 광고된 지원 여부, 실제 관측 값을 구분한다. `readyok`는
프로토콜 준비 응답이며 옵션 적용 값을 관측했다는 증거가 아니다. 미관측 값과 VRAM peak는
`unknown`으로 남긴다. 물리 NN 완료 입력 수·실행 수·실제 소비 입력 수·cache 소비 수와
P/C 작업·CPU 요청/완료/소비/노드 수를 구별한다. 외부 엔진의 내부 NN 실행을 RoveZero
계측값으로 꾸며 기록하지 않는다.
확인된 CPU affinity 불일치·메모리 한도 초과는 `ResourceAdmission` 실패로 보존한다.
UCI 옵션 이름의 대소문자 중복과 Ponder·Chess960·thread 한도·typed 모델 선택을
덮어쓰는 옵션을 거부한다. GPU 요청 모델명과 실제 UUID는 같은 표현이라고 가정하지 않는다.

CUDA 제품 시작은 runtime·모델·session의 cold 준비와, 준비된 backend에서 수행하는
P/C·제어 ACK probe를 분리한다. `--pals-startup-probe-timeout-ms`는 후자의 명시적
예산이며 범위는 1~180,000ms다. 생략하면 기존 15,000ms와 기존 JSON 생략 형식을
유지한다. V3 CUDA launch의 선택값을 시작·종료 execution에서 정확히 대조하고,
외부 `handshake_max_ms`는 cold 준비와 프로토콜을 포함한다. CPU CLI에는 이 CUDA
옵션을 허용하지 않는다. probe 예산이 handshake 이하여도 실제 cold 준비 여유가
보장되는 것은 아니며, 전체 pair 창·물리 drain·quarantine 조건은 늘리지 않는다.

시작 실패는 성공 readiness와 별도 게시 경로로 보존한다. 없는 mapping·placement
ACK는 없는 상태로 남기고, 관측된 ACK는 실제 소유 identity와 대조한다. 원래 오류·
부분 계수·논리 마감·미확정 물리 fence를 저장하며 실패한 시작을 성공 종료로 바꾸지
않는다. 진단은 최대 7개 제어 명령과 2개 역할 요청·64개 backend 단계만 기록한다.
lock 경합·poison·overflow는 진단 부재로 표시하며 실행 결과나 buffer 수명을 바꾸지
않는다. 명시 예산의 정상 실행에서는 실제 factory·probe 시간을 별도로 남기고,
일반 `go`에서는 startup 단계 기록을 닫는다. 단계 반환 사건 자체는 NN 완료 계수나
독립적인 물리 완료 증거가 아니다.

각 판의 백·흑 엔진, 실제 시계, 결과·종료 이유와 PGN hash를 보존한다. 엔진 크래시·불법 수·
시간패는 해당 엔진의 패배로 남긴다. 인프라 실패·실행 시간 한도·최대 ply 도달은 incomplete로
남기며 자동 무승부나 유효 대국으로 만들지 않는다. 불완전 영수증도 검증 가능한 증거로
보존하지만 `pair_eligible=true`를 붙이지 않는다.

## 단계별 인수

| 단계 | 구현·검증 대상 | 실제 학습과의 관계 |
|---|---|---|
| P0 | 공통 PALS 경계, search session 교체, 명세·lock·receipt, V1/V2 소비자 호환 | 학습 없음 |
| P1 | 자체 PVS CPU, 정확한 종료·TT·중단·조건부 resume, 독립 CPU UCI | 초기 평가와 학습 평가를 구별 |
| P2 | CPU/mock PALS 제안·반박·수선·결론 갱신, 유한 저장소·세대·수명 | mock 결과를 신경망 기력으로 해석하지 않음 |
| P3 | 384 폭·16 latent·고정 2회 반복 모델의 Rust 입력·실행·출력, export·독립 수치 대조 | 초기 파라미터 실행과 학습 완료를 구별 |
| P4 | 자체 데이터 계약, 분할·누출·unknown label, 비용·checkpoint·resume·export 도구 | 실제 optimizer 학습 실행은 이번 목표에서 제외 |
| P5 | V 학습용 작업·보상 계약, private 정보 분리, V-free export 검사 | 실제 V 학습은 실행하지 않음 |
| P6 | 자체 CPU 상대 pilot, 전체 시스템과 단일 축 효과 기록, 시계·PGN·종료 인수 | 강도 향상은 관측 결과로 판정 |

이 표는 완료 선언이 아니다. 총괄은 실제 구현 소스·같은 integration SHA의 CPU 검사,
신경망 수치 검사·GPU 물리 수명·대국 자료를 확인한 뒤 단계 상태를 갱신한다.
Colab의 학습 실행과 비용 등록은 별도 후속 실행이다. 원시 데이터·모델·PGN·로그·
보고서는 기존 저장소 관리 규약에 따라 저장소 밖에 두고, 이번 변경에는 소스와
검토 가능한 작은 독립 검사만 포함한다.

### 전체 원문과 최소 pilot의 구분

P6의 양성 자체 CPU pair만으로 두 원문의 전체 구현 완료를 선언하지 않는다.
현재 선택된 실행 흐름의 잔여와 원문의 선택적 후보를 다음처럼 구별한다.

| 항목 | 구현·인수 경계 |
|---|---|
| CPU_T SEE 정렬 | `LegalSeeV1` Rust ordering과 private CPU_T의 명시적 `--cpu-ordering=legal-see-v1`·봉인된 `ordering_policy: legal_see_v1`, producer의 `--ordering-policy legal_see_v1`까지 연결돼 있다. legacy 기본값·식별은 유지하며 SEE는 전략적 증명이나 영구 pruning이 아니다. 등록 기준 `b6de532`의 Linux·Windows CI에서 실제 선택 engine·조건·report·독립 resume namespace를 검사했다. 새 실제 producer 실행과 기력 효과는 별도 인수한다. |
| V 결과 환류 | bounded coverage loop와 native Repair 보고의 단일 prior 자료 발급 접점이 있다. capture의 loaded binary·원래 S/E/W·Rules·raw body·부모 transport 비용을 검사하지만, 자료 발급만으로 actual native witness·다음 Query admission·episode ledger·whole-action utility를 발급하지 않는다. coverage 증가는 전략적 유용성이나 학습 target의 증거가 아니다. |
| GPU record 공유 메모리 | host whole-input page와 record별 CUDA resident packing을 구분한다. 명시적 `registered-packing-v1` lane의 missing subset 인코딩·device join·불변 read view·물리 pin·예약·격리·CLI/arena 연결 소스가 있다. source/CPU 검사와 실제 target GPU의 수치·물리 완료·eviction·메모리 관측 인수는 분리한다. 전체 입력 cache 성공을 record별 GPU 공유의 증거로 쓰지 않는다. |
| Repair 이후 C 재검토 | 같은 line을 한 번 재검토하는 기존 lane과 실제 새 C continuation의 별도 S lane·다중 Reply 관측 접점이 있다. 원래 첫 수·Repair line/revision·첫 ongoing 상대 anchor·실제 dispatch/소비를 대조한다. 새 source의 실제 다중 Reply 수집 인수와 전략적 강도 효과는 별도로 남기며 다음 round의 root Propose로 대신하지 않는다. |
| 실제 CPU pair | 해당 integration source/binary와 시계·NN 소비·typed failed-go 0·Rules·PGN·process·저장·cleanup을 모두 인수한다. preparation 자기 보고나 CPU CI로 대체하지 않는다. |
| 선택적 후속 기능 | CUDA private Warm, paused search stack, set-associative TT, fast pruning과 대체 optimizer는 원문의 후보·미결정 범위를 보존한다. 기본 Fresh 구현의 필수 통과 조건으로 임의 확대하지 않는다. |

실제 학습은 이번 목표에서 제외한다. 2026-10-09 재개에서는 총괄과 서브에이전트가
소유 경계를 나눠 구현하며, GPU를 포함한 최종 실행은 총괄이 등록된 소스·바이너리·
자산·환경과 유한 예산을 확인한 뒤 인수한다. 과거 GPU 보류·실패 자료는 그대로 보존한다.
구현·CPU CI·실신경망·실제 대국·GPU 인수와 미지원 옵션을 각각 기록하며 미실행을
성공 또는 skip 인수로 채우지 않는다.

### 현재 소스의 최종 인수 순서

재개 감사 기준은 `b6de5325d47a417461823f4a92329287c0e5af09`이며 그 SHA의 CPU CI
성공을 후속 수정이나 실제 frozen NN 실행의 성공으로 소급하지 않는다. 아래 목록은
학습 제외 목표의 잔여 인수이며 구현 담당의 개별 완료와 최종 통합 인수를 구분한다.

1. 중단된 CPU 실행의 실제 owner·lease·scratch 상태를 확인하고 원래 실패 자료를
   보존한다. 최종 source·feature·바이너리와 자산을 새 실행에 등록한다.
2. 실제 registered frozen 4N CPU caller와 독립 native witness를 인수한다. 원래
   S/E/W, 입력 전 loaded binary 확인, raw·Rules·물리 완료·정리를 각각 확인한다.
3. 실제 결과를 다음 immutable Query에 소비하는 episode·prior ledger와 전체 action
   비용·utility 경계를 연결한다. 중복·순서 역전·늦은 결과·취소를 거부하고 unknown은
   masked로 보존한다. legacy receipt를 만들어 native 관측을 대신하지 않는다.
4. 같은 통합 소스에서 actual 다중 Reply의 tensor·dispatch·소비·Repair anchor/revision,
   분할·누출·영수증·저장을 인수한다. GPU 수치·공유 page·물리 수명 검사는 별도로 남긴다.
5. 같은 최종 source/binary의 자체 CPU 상대 한 쌍을 기존 120초+1초·최대 256 ply·
   전체 15분+정리 30초로 실행한다. 두 판의 백/흑·시계·NN 소비·typed failed-go·Rules·
   PGN·EOF/reap/group 종료·필수 최종 저장을 확인한다. 과거 source에 고정된 준비
   명세·binary를 새 source의 인수로 사용하지 않는다.
6. 원문의 선택된 필수 요구사항을 source·CPU 검사·실제 CPU·GPU·paired 자료에
   대조하고 정확한 최종 SHA의 CI와 PR 인수 기록을 갱신한다. 미지원 선택적 연구,
   실제 학습 제외와 관측되지 않은 값은 완료 범위와 함께 명시한다.
