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
