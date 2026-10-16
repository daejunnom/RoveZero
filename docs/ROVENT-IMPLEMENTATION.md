# Rovent 구현·인수

## 목표와 기준

시작 기준은 PR25·PR26이 통합된 `609819ded605795c9ff0653adad391e875b1c4b2`다.
`feature/rovent-reciprocal-nnuec`에서 별도 PR로 구현한다. 사용자 제공
`Rovent_Integrated_Handoff_20261016.zip`과 depth/mate 결정 메모는 설계 자료이며,
현재 소스·관측과 구별한다. 이 문서가 새 작업의 항목·배정·인수 기록을 연결한다.

이번 목표는 **학습 실행을 제외한 플랜 구현**이다. 학습 도구·recipe·checkpoint·자료
계약은 구현하되 optimizer update·본학습·대규모 수집을 실행하지 않는다. CPU/mock,
실제 CPU 수치, 로컬4050 정확성, 원격 비용·기력 비교의 증거는 따로 관리한다.
기존 미관측·NOT_READY는 새 문서나 녹색 CI로 완료 처리하지 않는다.

큰 체크포인트마다 총괄이 항목별 근거와 남은 소비자를 대조하여 사용자에게 알리고,
완료한 담당에게 다음 독립 항목을 배정한다. 작은 의미 단위의 커밋·push·Draft PR
공유는 전체 검증을 기다리지 않는다. 학습·원격 실행·병합을 자동으로 진행하지 않는다.

## M0 최소 계약

[rovent 검색 계약](../crates/rz-contracts/src/rovent.rs)은 별도
`rovent-search/0.1` envelope다. 기존 EvalRequest/EvalOutput·PALS0.1·P/C/V artifact
역할 번호와 V1~V5 기록을 바꾸지 않는다.

- `SearchRecipient`는 P/C/R 수신자이며 R은 `Role::Validator`의 별칭이 아니다.
- `ExecutionMode`의 Hybrid/CPU × P-C-R/P-C는 Collection/Deployment 목적과 독립이다.
  실제 backend와 native R 조립을 검사하며 조용한 CPU fallback이나 숨은 R은 허용하지 않는다.
- `QuestionEnvelope`은 원 root authority·parent·sender/recipient·정확 anchor·root-ply·
  원 first move·prefix/line/offset·view·질문·근거 revision·root budget·원 deadline을 보존한다.
  Store pin·Rules 검증·예산 ledger는 실제 manager의 소유다. digest가 capability를 대신하지 않는다.
- `SearchGoal`의 selective chess-ply depth와 mate move는 nodes·시계·NN 횟수와 독립이다.
  mate의 양쪽 horizon은 각각2N-1/2N이며 overflow와 등록 상한을 거부한다.
- typed framing 검사가 실제 질문 소비·합법성·물리 완료를 입증하지는 않는다.

## 확정된 depth·mate·root 제한

네 모드 모두 `searchmoves`, `depth`, `mate`를 정식 지원하는 것이 완료 조건이다.
parser만 받거나 원 driver가 무시하는 경로는 지원 완료가 아니다. legacy driver는
명시 capability가 없으면 새 목표를 검색 취소 전에 거부한다.

root 제한은 Rules의 합법 순서에서 한 번 봉인하고 P 제안·C 귀속·R root 작업·proof·
최종 선택·실패 fallback·ponder 보유 결과까지 전달한다. 제한은 root 첫 수에만 적용한다.
불법 또는 빈 명시 목록을 unrestricted나 체스 terminal로 바꾸지 않는다.

depth는 root-relative 선택적 chess-ply horizon이다. 실제 경계 평가와 선언된 후보·
반박·Repair 의무가 완료돼야 한다. 생성 길이·R 지역 depth·빈 큐를 완료로 대체하지 않는다.
전폭 또는1..D 전역 barrier를 필수로 두지 않고 시간제 go에 새 depth pass를 강제하지 않는다.

mate는 실제 Rules terminal에서 OR/AND로 전파한다. 한 유효 공격과 필요한 모든 합법
방어의 근거가 닫히면 강제 승리·패배 모두 조기 종료할 수 있다. 거리 상한과 정확 최단
거리를 구분하고 처음에는 정확 거리만 `score mate`로 출력한다. NN값·외부 mate·단일
협조적 PV는 증명이 아니다. 제한 root의 패배는 그 후보 집합에 조건부다.

P-C에서는 R/PVS/qsearch/SEE/외부teacher 호출이 실제0이어야 한다. 부족한 방어는 P/C
질문으로 처리한다. infinite/ponder의 계산 완료와 bestmove 출력 권한은 분리한다.

`go nodes`의 Rovent 단위는 `rovent-position-visits/1`이다. 실제 root 입장,
P/C 또는 mate 검토의 상태 전이, native R의 실제 방문 수를 합산한다. qsearch·SEE는
기존 native nodes의 부분이므로 다시 더하지 않는다. NN 호출·물리 입력 완료·PUCT
simulation으로 환산하지 않는다. 원 nodes 한도와 역할별 호출·native 상한을 각각
보존하며, 다른 엔진의 nodes와 같은 작업량이라고 해석하지 않는다.

## 구현 항목과 배정

진행 표는 source/소비/검사/실제 실행을 구분한다. 아래 첫 기록은 완료표가 아니다.
실행 원자료·장비 식별·개인 경로는 저장소 밖 보고서에 보존한다.

| ID | 책임 | 첫 구현/남은 인수 |
|---|---|---|
| RVH-E01 | 외부 연결 | 기존 SF 관측 재사용 조건 확인; Repair→C→Fresh 실제 소비 이월 |
| RVH-E02 | 외부 프로필 | 엔진·목적·옵션·자산·capability 분리; legacy guard 보존 |
| RVH-E03 | Reckless | 공식 pin·라이선스·실제 지원 제약 확인; query별 연결·종료 |
| RVH-E04 | 자료·외부 | raw CP/mate/PV·teacher와 내부 Fresh/Rules 근거 분리 |
| RVH-E05 | arena·프로필 | teacher/checker/opponent 목적과 전체 자원·시계·자산 기록 |
| RVH-A01 | 총괄 | 검색 envelope 게시; 실제 producer/consumer 연결 필요 |
| RVH-A02 | coordinator | 6방향 bounded 질문·합류·ancestor 대기·무진척·취소 |
| RVH-A03 | coordinator/R | anchor·prefix·응수별 frontier와 원 예산 소유 |
| RVH-A04 | CPU/R | 넓은 조사·지속 심화·R→P/C 실제 요청과 응답 소비 |
| RVH-A05 | CPU/runtime | paused owner admission 첫 변경; feedback 재개·정산 인수 |
| RVH-A06 | runtime | 미착수 local-first 분배·bounded 대기·결과 집계 |
| RVH-N01 | NNUEC Rust/참조 | 필수 명시 config·weights FP32 모듈; 독립 수치 대조 |
| RVH-N02 | NNUEC | B/C/L delta·실패 원자성·undo/full rebuild |
| RVH-N03 | NNUEC | anchor·수순 순서·승격·반박/복구 관계 보존 |
| RVH-N04 | CPU/NNUEC | 실제 조건부 value·후보 소비; 활성 전 N07 |
| RVH-N05 | 문맥·자료 | 보존·다음 계산 효용과 실제 결과/비용; heuristic 진단 분리 |
| RVH-N06 | 문맥 | raw/compact/view·착수 rebase·lease·가상토큰 중복 방지 |
| RVH-N07 | CPU/TT | history·model·C/L·조건 namespace·same-owner 재개 |
| RVH-M01 | 총괄/UCI | 네 모드 실제 조립·P-C R0·명시 backend |
| RVH-M02 | 총괄/runtime | 관리자·P/C·R·외부 실행의 전체 CPU/메모리·V-free 출전 |
| RVH-T01 | 자료 | 당시 입력/질문/문맥·출처 seal과 미래 target 분리 |
| RVH-T02 | 학습 도구 | 두 frozen actor·공통 학생/한 learner·checkpoint 후 교사별 fork |
| RVH-T03 | 자료·출전 | teacher/V 경험·V-free 경험·계보/전역 holdout |
| RVH-T04 | 학습 도구 | optimizer 소유·fork/resume·checkpoint/export 구현; 실제 학습 제외 |
| RVH-I01 | UCI | ponderhit 정상 시계 전환·miss/stop/root 교체·한 번 출력 |
| RVH-I02 | 총괄/UCI | Rovent 표시·artifact/호환 alias; 과거 domain 대량 치환 없음 |
| RVH-I03 | UCI/search | parser 첫 변경; root 제한·goal·horizon/proof·네 모드 인수 |
| RVH-V01 | 총괄 | 기존 원 근거의 영향 입력 대조; 미관측·NOT_READY 보존 |
| RVH-V02 | 담당별 | 새 기능의 실제 의미·오류·동시성·수명·수치 회귀 |
| RVH-V03 | 계측/arena | 기능 안정 후 별도 비용·동일 시간 대조; 구현 검사와 구별 |
| RVH-V04 | 총괄 | 체크포인트·잔여 배정·문서·Draft PR·최종 항목별 감사 |
| RVH-O01 | 조건부 후속 | 지역 커널 최적화는 병목 근거 후 한 변수 실험 |
| RVH-O02 | 조건부 후속 | 원본 외부 NNUE FT는 권리/필요성/학습 허용 후 별도 |
| RVH-O03 | 조건부 후속 | NUMA/다중GPU/원격은 실제 topology·장비 프로필별 |
| RVH-O04 | 조건부 후속 | compact 직접 입력·공동학습은 별도 모델/encoding |

## 통합 순서와 검증

첫 CPU 통합 체크포인트에서 확인한 결과는 아래와 같다. 실행 원본은 저장소 밖
`rovent-implementation-20261017-1` 보고서에 보존한다. 이 표는 작업 중인 소스의
로컬 검사이며 최종 통합 SHA·목표 GPU·성능·학습·대국 인수를 뜻하지 않는다.

| 실제 실행 | 결과와 범위 |
|---|---|
| 공통 계약 | 29개 통과 |
| search 라이브러리 | 723개 통과, 기존 선택 실행 2개 미실행; 최초 reciprocal·NNUEC·공통 depth/mate·동기 호환 owner 포함 |
| root 제한/batch 계약 | `experimental-batch` 조합 31개 통과 |
| NNUEC V1 독립 reference | Python 13개 통과; Rust caller와 별도 13입력 대조에서 raw/WDL 최대 차이 0 |
| 학습 준비 | neural import를 차단한 21개 통과; optimizer·tensor 복원·학습·실제 export는 미실행 |
| 새 명세와 CLI | 의미 검사 13개·CLI 9개 통과; 선언/저장 구조 검증이며 actual 실행 권한은 발급하지 않음 |

수치 대조는 명시적으로 만든 작은 미학습 FP32 파라미터만 사용했다. NNUEC V1의
정확 합산·B/C/L·조건별 TT 근거를 보존하며, 제안/반례/복구의 kind와 directed
parent/supersedes를 구별하는 입력은 **별도 V2 모델**로 추가한다. V1 자산을 V2로
자동 변환하지 않는다. 문맥 효용은 아직 실제 보존/경로 정책을 바꾸지 않는 관측이다.

상대 prefix 봉인, 공통 nodes 한도, native accepted-role checkpoint, 비동기 manager,
새 UCI owner와 early-terminal depth 영수증의 후속 변경은 위 최초 실행 이후에도
진행 중이다. 최종 소스를 고정한 뒤 영향 검사를 다시 실행해야 한다. native
selected-view 모델은 동기 호환 owner에서 명시적으로 거부하며, 실제 ticket facade와
R-free 조립이 연결되기 전 네 모드의 실제 신경망 지원 완료로 세지 않는다.

1. M0 계약·기준선과 UCI root 제한, NNUEC reference, 외부 프로필을 병렬 연결한다.
2. 기존 stage/worker/ledger를 재사용해 질문 그래프와 R-free 실제 경로를 만든다.
   suspended parent의 owner를 idle로 취급하거나 paused task를 다른 worker로 옮기지 않는다.
3. 같은 root-ply controller·Rules reducer로 네 모드의 depth/mate와 UCI 수명을 연결한다.
4. NNUEC의 명시 FP32 수치·delta·순서·undo를 대조한 뒤 context/local TT namespace를
   먼저 잠그고 실제 CPU 가치 소비를 활성화한다. 미확정 폭을 기본 모델 사양으로 고정하지 않는다.
5. 자료/학습 준비 consumer·다중teacher provenance·전체 계보 holdout을 연결한다.
   실제 학습을 하지 않아도 잘못된 resume/fork·state 변경·입력 누출은 검사한다.
6. 기능별 CPU·실제 NN/GPU·외부 owner 검사를 인수하고 남은 이월 항목을 따로 보고한다.

원 상호 질문·취소·정산·physical fence를 우회하는 테스트 전용 제품 경로는 추가하지 않는다.
일반 대국·학습·기력/성능 승격은 단위 검사나 구현 파일 존재로 대신하지 않는다.

## 두 번째 통합 체크포인트

후속 작업 중 search 746개가 통과했고 기존 선택 실행 2개는 미실행으로 남겼다.
이후 semantic Store/model과 R owner를 값 이동으로 분리한 소스는 제품 컴파일을
통과했으며 768개 중 764개 통과·fixture 2개 실패·기존 선택 실행 2개 미실행을
관측했다. 실패는 앙파상 FEN의 fullmove 번호와 실제 Proposal bundle이 없는 peer
재시도 fixture에서 발생했다. Rules·필수 bundle 검증은 유지하고 수정 후 재검사한다.

NNUEC V2는 별도의 미학습 FP32 모델로 등록했다. 네 record kind와 directed
parent/supersedes, 원 first move와 rebased suffix를 구별한다. 독립 Python 검사
13개, utility reference 10개와 Rust/Python 고정 입력 20개 대조를 통과했으며 후자의
raw/WDL 최대 절대 차이는 0이었다. 이 결과는 작은 합성 입력의 수치 동등성이고
실제 R evaluator 설치·GPU·기력·성능의 근거로 확대하지 않는다. utility는 계속
실행 권한과 known target을 발급하지 않는 shadow 관측이다.

새 corpus parser의 Rust 컴파일과 Python 계약 검사 14개, 학습 창·sampler의
표준 라이브러리 검사 5개도 통과했다. recovered wire 자료는 원 native/teacher
실행 권한을 발급하지 않는다. Torch forward/gradient·tensor 복원과 실제 trainer
실행은 이 로컬 결과에 포함하지 않았다. 이번 목표에서 실제 학습은 제외한다.

PR의 CPU 모델 CI에서 관측한 54개 실패는 모두 기존 fixture의 고정 Rust source
pin 변경 거부였다. 검증을 끄거나 실행 때 현재 hash를 자동 신뢰하지 않는다.
기존 closed source pair를 보존하고 실제 영향 경로를 검토한 새 조합을 별도로
등록한 뒤 같은 SHA의 CI로 재검사한다. bindings 성공과 나머지 CI 상태를 구분한다.

남은 핵심 연결은 실제 네 모드의 native 조립, R-free selected-view owner,
공통 node quota와 Candidate/Counter/Repair depth·Rules mate frontier, 실제
NNUEC 준비/worker 설치, 외부 UCI 실실행, typed 원자료·trainer/checkpoint 소비다.
각 접점의 source 준비와 실제 인수를 나누고 Draft PR에서 계속 공유한다.

후속 수정·module 등록 뒤 search 783개 중 **781개 통과·기존 선택 실행 2개
미실행**, experiments 234개 통과를 확인했다. 여기에는 semantic/R 필드 분리,
실제 accepted-source publication과 first move 변조 거부, Frontier의 typed 원
질문·Fresh/Rules·lease 한도, 공통 원장 영수증, NNUEC V2 CPU/utility 검사가
포함된다. 새 worker evaluator 설치는 이때 제품 컴파일만 확인했으며 설치 전용
수명 회귀와 상위 실제 job 연결은 다음 인수다. search 구현 digest에 새 Frontier
소스를 포함한 provenance 갱신도 별도로 기록한다.

## 세 번째 통합 체크포인트

이후 작업 중인 소스에서 search **798개 통과·기존 선택 실행 2개 미실행**,
`pals-adaptive,onnx-cpu` UCI 라이브러리 **604개 통과**를 관측했다. P-C 전용
semantic port, 독립 pending 질문, 설치 전용 evaluator 수명, typed Frontier
입력과 원 종료 오류의 보존이 포함된다. 이 checkpoint 뒤에도 PC의 실제
Frontier 소비·NNUEC 최초 준비·정상 예산 종료를 연결하고 있으므로 이 결과를
최종 소스나 네 모드의 신경망 종단 인수로 재사용하지 않는다.

표준 라이브러리 기반 준비·corpus 계약 검사는 Windows에서 **81개 통과·Linux
전용 9개 건너뜀**, WSL Ubuntu에서는 **90개 모두 통과**했다. 이 중 새 학습
도구의 순수 계약 검사는 48개다. 모델·optimizer 실행을 차단한 검사이며,
실제 학습·Tensor 복구·수치 export·GPU 성공을 의미하지 않는다. 별도로 준비한
PyTorch forward/gradient·Tensor 수명 검사 20개는 로컬 Torch 부재로 미실행이다.

새 `rz-rovent-source-aggregate/1`은 원 commit과 정렬된 component map을 길이
프레임으로 묶는다. `rovent_source_aggregate_sha256`의 독립 Python 고정값 대조와
입력 한도/식별 거부 검사 두 개가 통과했다. 이 값은 선언한 source set의 식별이며
compiled/pinned bytes 대조나 실행 권한을 대신하지 않는다. corpus의 runtime
식별은 실제 loaded runtime library의 SHA-256이며, 전체 runtime 구성의 집계로
해석하지 않는다. 원 bundle·backend·native configuration·lock은 별도로 보존한다.

원격 `3a0be9f` CI의 Linux/Windows 실패는 Ponder 활성화 fixture가 합법적인
예측 착수를 포함한 출력을 거부한 사례였다. 원 제한 착수와 상대의 합법적인
예측 착수를 따로 확인하도록 수정한 뒤 위 로컬 604개가 통과했다. 독립 bindings
lock의 누락된 `memoffset` 의존 항목도 package version 변경 없이 동기화했다.
모델 CI의 source pin 거부와 최종 SHA의 CI 인수는 여전히 별도이며, 자동 hash
신뢰나 기존 receipt 재서명으로 해결하지 않는다.

Linux 전용 수집·학습 parent의 첫 교차 컴파일에서는 collector 접점 네 곳의
타입/가시성 불일치를 관측했다. 이력은 보존하고 실제 NNUEC module·원 native
입력 key·공개 SHA 경계로 수정한 뒤 같은 feature의 Linux 대상 lib/tests 교차
컴파일이 통과했다. 교차 컴파일 통과 자체도 Linux
자원 적용·실제 child·외부 ELF·GPU·학습 성공과 구별한다.

다음 병렬 인수는 실제 네 모드 launcher와 startup 설정, 원 질문에 묶인
Candidate/Counter/Repair·mate missing queue, 동일 preparation actor의 최초
NNUEC R 준비, 원 수집 cap과 trainer parent, 외부 UCI 실제 실행이다. 이후
CPU/비학습 Tensor·목표 GPU 수치·물리 수명을 확인한다. 실제 학습은 이번 목표에서
제외하며 Draft PR의 병합도 이 checkpoint에 포함하지 않는다.

학습 도구의 추가 소스 감사에서는 같은 부모가 보유한 live corpus/window custody로만
재개할 수 있고, 부모 재시작 뒤 durable dataset/checkpoint를 소비하는 제품 경로가
없음을 확인했다. 이를 T04 미완료로 보존한다. 후속은 caller가 exact pin한 원
publication과 전체 rows/Rules/target/계보를 재검증하는 historical dataset 자격,
checkpoint 복구 자격과 새 trainer 호출부를 연결한다. 과거 native 실행·물리 완료
권한은 재발급하지 않으며, 미회수 실행과 checkpoint 이후의 비용 unknown도 유지한다.

다음 고정 소스의 추가 통합에서는 search **812개 통과·선택 실행 2개 미실행**,
UCI **607개 통과**를 관측했다. 최초 NNUEC 준비의 회귀 13개에는 동일 actor의
실제 handoff·미인수 엔진 회수·다른 actor 거부·실제 Ready 경계가 포함된다.
이는 작은 미학습 CPU 모델과 소유권 검사이며 전체 launcher의 실제 NN 인수가 아니다.
초기 node 0/1·제한 child·Rules-only mate와 원 cleanup ceiling도 검사했다.

독립 소스 감사에서 아직 인수하지 않은 두 실제 연결도 확인했다. P-C의 missing
defense와 accepted selection은 원 controller에서 먼저 청구한 checked snapshot을
Store에 채택해야 하며, 그 이전 Rules 전이를 허용하지 않도록 보완한다. 학습
window의 기존 단일 runtime 필드는 collection ORT library와 trainer Python
executable을 같은 값으로 요구하므로 새 명시 codec에서 분리한다. 같은 가짜 SHA를
쓴 fixture 통과를 실제 실행 가능성으로 해석하지 않는다.
