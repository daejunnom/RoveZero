# 결정·미결정 기록

최종 갱신일은 2026-10-03다. 초기 문서 작성일은 2026-10-02다. 사용자 선택,
이 문서의 도입 기본값, 핸드오프의
연구 제안과 실제 구현·관측 상태를 구분한다. 원문 기록은
[reference](reference/README.md)에 보존하고 실제 작업은
[IMPLEMENTATION-DIRECTIVES](IMPLEMENTATION-DIRECTIVES.md)를 따른다.

## 기록의 권한

- 초기 요청은 기존 규약을 구현 지시서 수준으로 강화하고 공동작업의 WIP 커밋·
  push를 늘리며 지정한 20개 작업·총괄의 최소 공통 계약 책임을 반영하는 것이다.
  가중치의 권리와 RTX 4050 6GB 적합성 조사 후 한 종류를 선정하라는 요청도 포함한다.
- 후속 요청은 기존 의미를 유지한 공용 규약을 총괄·A~F에 적용하고
  `daejunnom/RoveZero` 원격에 추가하는 것이다. 이 준비 채팅의 총괄 배정과 이후
  목표 설정·일시중지·사용자 재개 절차는 D021/D026/D027, 빈 원격의 최초 Git 구성은
  D028로 구분한다. 문서를 읽는 다른 에이전트에게 총괄 역할이나 일시중지를 자동 부여하지 않는다.
- 핸드오프의 D01~D12·V 표·17.4절은 당시 요청·검토 기록이다. 원문의 `확정`을
  이번 세션에서 사용자가 새로 승인한 구현 선택이라고 재표기하지 않는다.
- 아래의 `사용자 선택`은 이번 대화에서 직접 지정한 내용이며 `도입 기본값`은
  변경 가능한 구현 기준이다. `연구 제안`은 최종 구현 채택이 아니다. 변경 시
  이유·출처·검증·영향을 기록한다. 독립 Rust 엔진·신규 자체 MIT 코드·자체 코어·
  최소 PUCT는 사용자 선택이다. 모델·backend·훈련 언어·실행 예산은 별도 결정한다.
  문서나 조사 완료가 엔진·학습·대국·원격 변경의 실행 완료를 뜻하지 않는다.

## 이번 작업에 반영한 기준

| ID | 상태 | 기준과 근거 |
|---|---|---|
| RZ-D001 | 현재 사용자 요청 | Accelerate 규약을 참고해 로컬 RoveZero 규약을 설정하고 구현 지시서 수준으로 강화한다. Accelerate는 수정 대상이 아니다. |
| RZ-D002 | 설계 자료 기준 | v0.2.0을 중심으로 읽고 14~17장의 개정·측정 원칙을 우선한다. v0.1.0과 두 원문은 무변경 이력으로 보존한다. 현재 사용자의 구체 결정이 설계 제안보다 우선한다. |
| RZ-D003 | 설계 기준 | 표준 체스의 LC0 대비 동일 장비·시간 득점률을 목표로 한다. Augment Chess 전용 계약은 옮기지 않는다. |
| RZ-D004 | 도입 기본값 | 한국어 문서, 읽기 전용 시작 조사, 기존 작업 보존, 같은 목표의 브랜치·연구 기록 재사용. |
| RZ-D005 | 도입 기본값 | Git 도입 후 `develop` → `feature/*` → `develop` PR. 실제 Git·보호 설정을 먼저 확인한다. |
| RZ-D006 | 도입 기본값 | SRP는 기능·변경 이유·소유권 기준. 줄 수 제한·실험별 복제·불필요한 빈 폴더를 두지 않는다. |
| RZ-D007 | 도입 기본값 | 외부 입력·오류·버전·정밀도·소유권을 경계에서 검증한다. 문서·CPU/mock·수치 참조·실제 GPU·강도·배포 증거를 구분한다. |
| RZ-D008 | 도입 기본값·환경별 경로 | Windows 생성물은 저장소 밖 `%APPDATA%\RoveZero`, CI는 `$RUNNER_TEMP/RoveZero`를 사용한다. Windows 호스트의 WSL 보존 산출물은 확인한 Windows 루트로 내보낸다. 독립 Linux 클라우드는 Windows 경로를 요구하지 않고 저장소 밖 작업 전용 output root·소유자·보존 기간·작업 종료 전 회수 경로를 명시한다. 재생성 캐시와 보존 근거는 구분한다. |
| RZ-D009 | 도입 기본값 | 보존할 모델·데이터·PGN·보고서는 재생성 캐시와 구분한다. 검토된 요약·작은 fixture·설정만 소스에 둔다. |
| RZ-D010 | 사용자 경로로 갱신 | 초기 `network/` 제안은 사용자 지정 `crates/rz-encoding/`·`crates/rz-eval/`로 대체한다. `models/`는 생성 가중치·접근 제외 영역이다. 훈련 도구 경로·언어는 후속 결정한다. |
| RZ-D011 | 연구 분류 기본안 | G 주 방향, H 보조, C 비교, O 오프라인. 분류를 적용하되 최종 구현 배치는 미결정이다. |
| RZ-D012 | 정확성 기준 | 자체 Rules/State의 단일 권한, 정확 캐시·근사 잠재 상태·raw/corrected 평가·edge 통계 분리. 조회와 새로운 유효 traversal의 backup을 구분한다. |
| RZ-D013 | 평가 계약 기준 | 동일한 완전 시작 상태에서 엔진의 흑백 배정을 교환한다. T1 전체 자기 시계·같은 자원 상한이 주 평가다. |
| RZ-D014 | 평가 계약 기준 | Ponder·상대 차례 계산·온라인 가중치 변경 금지. 게임 안 history/correction은 명시 옵션·자기 시간·newgame 초기화로 구별한다. |
| RZ-D015 | 연구 보존 기준 | 76개 후보의 원문 ID·출처·기각·보류·부정적 결과를 보존한다. 하드웨어 적합성 추론을 측정 성과로 바꾸지 않는다. [등록부](CANDIDATE-REGISTER.md)에 구현·검증·중단을 구체화한다. |
| RZ-D016 | 사용자 순서로 갱신 | 총괄 계약 → A·B·C01·D01·E·F01 병렬 → C02/C03 실제 Rust 기준 → D02 계측 → 새 탐색 또는 D03 runtime → F02 → F03. R0~R7은 연구 단계로 보존하며 LC0 runner 개발은 별도 병렬 가능하다. |
| RZ-D017 | 사용자 요청 반영 / 간격은 도입 기본값 | 공동작업을 위해 작은 진행 단위마다 커밋·push한다. PR 커밋 수 상한을 없애고 미완성·빌드/테스트 실패도 WIP로 공유한다. 장기 작업은 15~30분 간격으로 변경을 확인하며, 실제 상태를 명시하고 통합·완료 검증은 별도로 충족한다. 기존 1~3개 커밋·최종 1회 push·60~90분 완료 단위 기준을 대체한다. |
| RZ-D018 | 사용자 선택 | 독립 Rust workspace와 자체 체스 코어. 외부 체스 라이브러리는 독립 perft·상태 대조용이며 제품 핵심 구현을 대신하지 않는다. LC0 fork를 첫 제품 기반으로 삼지 않는다. |
| RZ-D019 | 사용자 선택 | 신규 자체 코드 MIT 방침. 외부 코드·가중치·데이터·도구·opening의 권리는 각각 확인하고 MIT로 일괄 재표기하지 않는다. 실제 LICENSE 파일·의존 검토는 소스 도입 시 처리한다. |
| RZ-D020 | 사용자 선택 | 최소 자체 PUCT는 정확한 선택·확장·backup·방문·terminal을 검증할 기준선이다. 교체 경계를 두며 최종 탐색 알고리즘으로 고정하지 않는다. |
| RZ-D021 | 사용자 선택·초기 배정 이력 | 초기 준비 대화의 에이전트를 총괄 TASK-I01/I02와 최소 공통 계약·`rz-contracts`의 단일 소유자로 배정했다. 공용 규약에서 총괄의 의미는 이 역할로 배정된 에이전트이며 독자 모두를 뜻하지 않는다. A03은 Position 적용·검증 담당이다. 공통 변경은 총괄의 영향 검토·revision 갱신으로 통합한다. |
| RZ-D022 | 사용자 선택 | 20개 담당 작업·crate 소유·인수 조건은 구현 지시서에 따른다. 담당 `TASK-*`와 76개 연구 `CARD-*`를 별도 namespace로 기록한다. |
| RZ-D023 | 사용자 선택 | 내부 대조는 CONTROL-0(W0/S0) → CONTROL-1(W0/S1) → CONTROL-2(W1/S1). D03 runtime은 같은 W·S의 별도 축이다. LC0 대국은 외부 성능 비교이며 변경 원인의 내부 A/B와 구분한다. |
| RZ-D024 | 사용자 요청에 따른 조사·선정 | 첫 호환 가중치는 Maia1 v1.0 `maia-1900.pb.gz`로 선정했다. 원저자의 GPL 적용 근거·6×64 SE/WDL 형식을 확인한 인간 수 예측 모델의 호환·runtime 기준선이다. [WEIGHT-SELECTION](WEIGHT-SELECTION.md)의 digest·권리·형식·RTX 4050 자료상 적합성과 C03 실제 인수를 따른다. T70는 권리 미확인으로 보류하며 최종 강도 모델과 실제 배포 의무 충족은 별도다. |
| RZ-D025 | 계약 도입 기본안 | [CONTRACTS](CONTRACTS.md)의 side-to-move WDL, a1=0 square, typed IDs·generation·deadline·수명·유한 예산을 최초 의미 계약으로 사용한다. 실제 Rust 표현·backend 오차는 해당 연결 전에 잠근다. |
| RZ-D026 | 사용자 요청 / 2026-10-03 | 기존 규약의 Rust·MIT·자체 코어·최소 PUCT·총괄 계약 소유권·A03 적용·WIP 공유·보호·증거 의미를 유지하며 총괄·A~F가 함께 사용한다. 각 에이전트는 사용자 배정에서 역할·TASK ID·소유 경로를 확인한다. 공용 규약을 읽는 행위는 총괄 배정이나 실제 구현·실험 권한의 추가가 아니다. 공유 규약과 필수 참조 문서·보존 원문을 원격 `daejunnom/RoveZero`에 추가한다. |
| RZ-D027 | 사용자 요청 / 이번 준비 채팅의 총괄 절차 | 원격 추가를 마치고 담당 진행 순서를 안내한 뒤 사용자가 이 총괄 채팅의 목표를 설정하면 즉시 그 목표를 일시중지한다. 사용자가 명시적으로 재개할 때마다 관련 원격 PR의 상태·head/base SHA·변경 파일·CI·리뷰를 먼저 확인하고 총괄 범위에서 진행한다. CI 요청·PR 존재·대기를 통과나 사용자 승인으로 보지 않는다. 이 채팅에 대한 일시중지 요청을 다른 A~F 에이전트나 다른 목표에 자동 전파하지 않는다. |
| RZ-D028 | 최초 빈 원격의 bootstrap 예외 / 2026-10-03 | 기존 heads/tags와 PR이 없는 원격을 확인한 뒤, 최소 `.gitignore`만 `main`의 첫 커밋으로 공유하고 같은 SHA에서 `develop`을 만든다. 공용 AGENTS·필수 참조 문서·두 원문 snapshot은 `feature/docs-shared-agent-rules`에서 `develop` 대상 PR로 추가한다. 이 최초 브랜치 구성만 D005의 일반 흐름 예외이며 이후 작업은 feature PR을 따른다. 보호 규칙을 우회하거나 기존 공유 이력을 재작성하지 않는다. 실제 bootstrap SHA·PR 번호·체크 상태는 Git·원격에서 조회해 보고하며 이 결정으로 실행 성공을 선기록하지 않는다. |
| RZ-D029 | 사용자 환경 지정 / 2026-10-03 | A~F는 GPU가 없을 수 있는 클라우드에서 개발한다. CPU/mock 기본 조합·가능한 CPU 신경망 참조·backend 코드·fixture·검사 진입점·학습 recipe/resume 검사는 먼저 진행할 수 있다. 목표 RTX 4050 6GB의 실제 GPU 추론·D02 계측·D03 GPU 개선·정식 대국과 실제 학습은 지원 환경에서 총괄 I02가 별도로 인수한다. CUDA capability 부재와 GPU 검사 skip·미실행을 기록하며 조용한 CPU 대체·CPU/mock CI의 GPU 통과 승격을 금지한다. 코드 완료·실제 학습·강도 인수는 구분한다. 기존 사용자 배정의 실행·자원·비용 범위는 재승인 없이 진행하고 이 규약 공유만으로 새 비용·범위를 추가하지 않는다. 독립 클라우드 생성물은 D008의 작업 전용 루트·보존·회수 조건을 따른다. |
| RZ-D030 | 사용자 직접 허가 / 이번 문서 작업만 | 이번 공용 규약·구현 지시서 공유 작업에 한해 `main` fast-forward를 허용한다. `develop` 대상 feature PR로 변경을 남기고 문서·원문 무결성 검증을 마친 동일 SHA를 `develop`과 `main`에 fast-forward로 반영한다. 공유 이력 재작성·force-push를 하지 않는다. 이후 구현은 D005의 일반 feature PR 흐름을 따른다. 실제 반영 SHA·PR 상태는 원격에서 확인해 보고한다. |
| RZ-D031 | 총괄의 최초 Rust 계약 / 2026-10-03 | `rz-contracts` 0.1.0, SchemaVersion 0.1 exact match를 게시한다. std-only typed IDs·epoch·immutable generic snapshot·ordered legal view·모델/인코딩 descriptor·WDL/합법 policy·단조 ns deadline·취소·fresh 평가 요청/결과·오류/유한 예산을 최소 경계로 구현한다. 구체 규칙·탐색·backend·runtime 및 wire/hash codec은 consumer 소유이며 계약 crate 자체 검사와 실제 연결 인수는 구분한다. |
| RZ-D032 | 사용자 직접 요청 / 2026-10-03 | 다른 담당의 Draft PR 계약 요청을 조사하고 공통 계약을 만든 뒤 해당 PR에 적용 코멘트를 남긴다. PR 인수 때 총괄은 최신 코드의 adapter·루트 Cargo/lock·revision·시간·오류·policy·수명 경계를 수동으로 맞추고 같은 integration SHA의 소비자 검사를 수행한다. 계약 게시만으로 소비자 통합·실제 NN/GPU 완료를 보고하지 않는다. 이번 요청은 일시중지한 후속 목표 앞의 선행 계약 작업이며 D030의 과거 main 예외를 확대하지 않는다. |
| RZ-D033 | 사용자 후속 환경 지정 / 2026-10-03 | 후속 GPU 벤치마크는 로컬 대신 외부 환경에서 진행한다. 서브에이전트가 RunPod 공식 자료를 조사하고 총괄은 로그인된 ChatGPT 브라우저 콘솔의 공개 배포 후보를 읽기 전용으로 확인해 [준비 계획](research/RUNPOD-BENCHMARK-PLAN.md)을 작성한다. 사용자가 이번 응답 이후 GPU를 지정하므로 후보 선택·지역·실행 시간·총액·보존 조건은 미결정이다. 이 계획은 Pod 생성·유료 실행·새 credential/접근 확장의 승인이 아니다. 기존 RTX 4050 인수 이력과 외부 GPU 측정은 구분하고 내부 A/B·LC0 비교는 같은 지정 장비·조건으로 다시 잠근다. |
| RZ-D034 | 사용자 후속 지정·CPU 최적화 검증 / 2026-10-03 | GPU A/B 전에 의미 보존 중복·자료 복제·병목을 조사·해결한다. `1cdd707`에서 Rules export/부모 fork/preview delta/불변 profile hash의 중복을 줄이고 상태 witness·CPU 교차 측정·양 OS CI를 확인했다. Community 4090 등 live 후보를 조회하며 network volume을 사용하지 않고 초기 총 지출 가능액 약 US$200 안에서 유한 하위 예산을 둔다. 사용자 지정 Oracle에는 작은 원본 연구 자료와 provenance를 검증해 보존한다. key 내용은 읽지 않고 지정 SSH 인증에만 사용하며 Pod Env 등록은 사용자 담당이다. 실제 Pod·GPU A/B는 미실행이고 GPU 선택·환경/명세·과금 종료 조건은 실행 전 잠근다. [사전 최적화](research/PRE-RUNPOD-OPTIMIZATION.md), [외부 계획](research/RUNPOD-BENCHMARK-PLAN.md)을 따른다. |
| RZ-D035 | 사용자 후속 요청·총괄 조사 / 2026-10-04 | Maia의 강도 한계를 검토하고 로컬 LC0 실행과 모델 교체/추가 학습 계획을 준비한다. 공식 LC0 v0.32.1 Windows CUDA package의 digest, Maia와 동결 T1 distilled의 실제 로컬 GPU smoke, T1 원본 구조/직접 제작자 허가를 확인한다. 현재 Rust Maia exact profile과 계약 0.1은 유지한다. 총괄 권고는 B/D의 WorkerLimit·물리 완료·128 simulation 상한을 별도 검증하고 C의 동결 강도용 모델 호환을 먼저 인수한 뒤 F02를 진행하는 순서다. 이번 조사는 모델 승격·실제 학습·새 유료 GPU·정식 강도 성공을 뜻하지 않는다. [전환 계획](research/LOCAL-MODEL-BASELINE.md)을 따른다. |
| RZ-D036 | 사용자 BT4 실제 적용·벤치마크 지정 / 2026-10-04 | BT4-it332 단일 원본을 별도 source/export/model profile로 Rust CPU/CUDA에 연결하고 같은 원본의 native LC0 FP32·RoveZero FP32·별도 LC0 FP16 로컬 위치 벤치마크와 개발 pair를 실행한다. Maia exact profile·공통 계약 0.1·자체 Rules/PUCT를 유지한다. BT4 개별 license는 미확인·재배포 false로 보존하고 실제 학습은 시작하지 않는다. 기본 128 simulation은 유지하며 native에 유한 1..4096 선택을 추가한다. raw 실패·시간/물리 완료 공백·즉시 메이트 선택 한계를 보존하고 개발 표본을 정식 강도/Elo/훈련 인수로 승격하지 않는다. [실행 기록](research/LOCAL-MODEL-BASELINE.md)을 따른다. |
| RZ-D037 | 사용자 종료 회귀·조건 일치·B1 계측 지정 / 2026-10-05 | 메이트 후보→실제 Rules 종료→관점 backup→최종 착수를 먼저 추적하고, 모델/hash·이력·입력·합법 policy·WDL 및 동일 바이너리 A/A 뒤 host 병목을 검사한다. 이전 BT4의 explicit simulations=4096을 기본 128 문제로 재해석하지 않는다. LC0의 정확한 terminal/bounds와 신경망 추정치를 구분하는 정책을 검토해, 실제 승인된 직접 자식 terminal만 우선하는 opt-in `exact-terminal-child-v1`을 독립 S 변경으로 구현한다. 기존 방문 수 우선 S0·기본값과 공통 revision 0.1은 유지하며 NN Q=±1·단일 winning descendant·미방문/거부된 완료는 증명으로 사용하지 않는다. 전체 solved bounds 전파·mate-distance solver·기력 승격은 별도 후속 인수다. [회귀·계측 기록](research/LOCAL-MODEL-BASELINE.md#8-종료-회귀조건-대조b1-계측과-별도-s1)을 따른다. |
| RZ-D038 | 사용자 런타임 저장 공간 개선 / 2026-10-05 | 실행별 ORT/CUDA 전체 복제는 기본 bootstrap에서 해시별 공유 런타임 캐시로 대체한다. 이 불변 vendor 라이브러리만 OS/architecture/content identity로 checkout 간 공유하는 고정 슬롯 예외다. 가중치·모델·데이터·실행 로그는 이 캐시에 넣지 않는다. 최대 4 entry/8 GiB, hit의 전체 byte/readonly pin 검사, 직렬·원자적 publication, 손상 거부와 실패 staging 정리를 적용하고 활성 entry를 자동 교체·삭제하지 않는다. C가 저장 권한을 소유하고 B의 native CLI·수치 gate·E의 cleared child argv가 소비한다. 모델 평가 cache provenance·공통 revision 0.1·CUDA exact profile·process lifetime·기존 탐색 정책은 유지한다. [저장 공간·재사용 인수](research/PERFORMANCE-OPTIMIZATION-PLAN.md#13-실행별-native-library-복제-제거와-검증된-공유-저장)를 따른다. |
| RZ-D039 | 사용자 후속 진행 / 2026-10-05 | BT4 대국 후속에 앞서 B03의 독립 stop/deadline 출력 경계를 인수한다. 논리 취소 시 마지막 유효 착수를 고정하고 소유 worker의 정상 물리 drain·join 뒤에만 출력한다. 입력/isready는 유지하며 scope 교체는 보류 착수를 폐기한다. 같은 유한 shutdown_limit의 오류·panic·시간 초과는 원래 typed serve 오류로 전달하고 착수를 승인하지 않는다. 공통 revision 0.1·PUCT·WDL·S0/S1·cache provenance는 유지한다. 실제 CUDA 수명 질의와 고정 종료 회귀를 E BT4 artifact/profile·holdout 강도·device profiling 인수와 구분한다. [출력 경계 인수](research/LOCAL-MODEL-BASELINE.md#9-stop마감-출력의-물리-완료-경계)를 따른다. |
| RZ-D040 | 사용자 후속 진행 / 2026-10-05 | E에 BT4 CUDA launch V2의 별도 schema/domain·exact source·ONNX/arena 한도와 실제 search-config receipt 대조를 추가한다. CPU/CUDA launch V1·공통 revision 0.1·provider V1·PUCT/S0/S1을 유지한다. 초기 second-model allocation 실패와 RAM 12/14 GiB 진단은 보존하며, verified private snapshot만의 캐시 힌트 뒤 같은 12 GiB에서 실제 E/B/C/D/A A/A 연결을 인수했다. 네 fresh process·CUDA/backup/drain·전체 PGN의 감사와 두 OS CI를 확인했다. 18 ply cutoff 두 판은 Incomplete·점수 대상 0·strength false이며 정식 시계·S0/S1 holdout·LC0 기력·학습 인수가 아니다. [V2 연결 인수](research/LOCAL-MODEL-BASELINE.md#10-e-bt4-cuda-v2의-실제-aa-연결-인수)를 따른다. |

## BT4 pilot 후속 결정 — 2026-10-05

| ID | 상태 | 기준과 근거 |
|---|---|---|
| RZ-D041 | 사용자 표본 확정·별도 탐색 정책 pilot | 같은 BT4 FP32/B1·PUCT에서 S0 visits와 S1 exact-terminal의 최종 선택만 비교한다. 사용자 확정은 16쌍·32판·전체 최대 120분, 시계는 30초+제시간 착수 뒤 0.1초·최대 256 ply다. 별도 CUDA V3 schema/domain·고정 cohort·흑백 교환·전체 position→bestmove 시계·자동 claim의 A 검증·고정 paired Hoeffding95를 잠근다. V2 A/A cutoff를 점수로 재해석하지 않으며 작은 pilot을 Elo·승격으로 보고하지 않는다. 실패·미완료·시작 전 엔진 실패도 원장에 남긴다. [pilot 명세와 인수](research/BT4-FINAL-SELECTION-PILOT.md)를 따른다. |
| RZ-D042 | 사용자 할당 실패 확인·개선·계속 요청 | 기존 RAM 압박과 이번 모델 초기화 할당 실패를 구분한다. RAM 최고치·한도 사건·GPU 표본과 별도 dual 진단만으로 물리 VRAM 부족을 확정하지 않는다. CUDA arena 확장을 요청 크기로 바꾸고 backend identity를 분리한다. BT4 독립 수치·Rules No/Repeat·두 엔진 상주 메모리·물리 종료·CPU 검사/CI를 먼저 인수한다. 실패한 pilot은 보존하고 새 source/binary/backend를 잠근 16쌍을 처음부터 실행한다. 두 시도의 점수를 합치지 않고 최초 전체 120분 안에서 진단·재실행을 마감한다. 모델·FP32·TF32 off·PUCT·S0/S1·arena 상한은 유지한다. |
| RZ-D043 | 사용자 전체 시계·피셔·엔진 blitz 기준·PGN 색별 엔진 종류·후속 진행 지정 / 2026-10-05 | 공식 CCRL Blitz의 2분+1초 시간 형식을 참조해 로컬 RTX 4050에서 새 paired pilot을 준비한다. 기존 30초+0.1초 결과를 보존하고 점수를 합치지 않는다. 같은 BT4/FP32/No/B1·PUCT·cap 4096의 S0/S1 16쌍·32판을 별도로 잠그며 후속 실행 전체 상한은 120분이다. 기준 Intel i7-4770K CPU 보정·EGTB·공식 CCRL 등록/rating은 수행한 것으로 표시하지 않는다. PGN TimeControl을 실제 부모 clock trace/잠금과 대조하고 raw White/Black에 RoveZero·BT4·S0/S1을 명시한다. SAN 검토 사본은 원본 색별 ID·binary/model/source hash·program type을 보존한다. 실패·미완료·호스트 RAM 압박·실제 인수를 기록하며 미관측 opening/기력 승격을 주장하지 않는다. |

## 남은 결정과 실행 전 잠금

사용자 후속 지시 RZ-D044: PGN TimeControl 차이는 진단 기록으로 보존하고 벤치마크를
차단하지 않는다. 시계도 두 상수의 허용 목록으로 제한하지 않는다. 이번 실행의 실제
값은 120초+1초이며 실제 부모 시계·엔진 색 배정·출력·실패는 그대로 관측한다.

기존 ID를 유지한다. `부분 결정`인 행의 확정 부분을 다시 미정으로 취급하지 않는다.

| ID | 상태·남은 항목 | 결정할 시점·필요 근거 |
|---|---|---|
| RZ-O001 | 부분 결정: 원격 RoveZero; 이름 권리 남음 | 폴더명 RoveZero를 사용하며 원격 대상은 `daejunnom/RoveZero`로 정했다. 빈 원격의 최초 구성과 규약 추가는 D028을 따른다. 원격명 확보와 이름·외부 자산의 권리 확인은 구분한다. |
| RZ-O002 | 부분 결정: 독립/MIT; 외부 코드·자산 조건 남음 | LC0 fork 여부와 신규 자체 코드 방침은 D018/D019로 결정했다. 실제 의존·도구·가중치·데이터의 사용·변환·학습·재배포 조건을 도입 전에 확인한다. |
| RZ-O003 | 부분 결정: Rust engine·첫 호환 format; backend·훈련 스택 남음 | Rust workspace와 Maia1 format은 결정했다. toolchain·target·feature·binding·변환/실행 backend와 훈련 언어는 장비·빌드 조건으로 잠근다. ORT/tract/PyO3를 자동 상속하지 않는다. |
| RZ-O004 | 개별 G/H/C 배치와 첫 측정 변경 | D02 근거로 새 탐색 또는 D03 단일 runtime 변경을 고른다. 모든 계산의 GPU 실행·CPU 증분 우선은 전제하지 않는다. |
| RZ-O005 | LC0 비교 identity와 첫 호환 가중치 인수 | version·binary/weights digest·search/backend/precision/options를 잠근다. 선정 조사의 권리·형식·자료상 GPU 적합성과 C02/C03 실제 수치·장비 검증은 분리한다. |
| RZ-O006 | 부분 결정: Community 후보·network volume 미사용·초기 총예산 약 US$200·Oracle 회수; 실제 GPU/환경 명세 남음 | A~F CPU 개발과 별도 실제 검증을 구분하고 초기 RTX 4050 인수 이력은 보존한다. D033/D034와 외부 계획에 따라 사전 CPU 최적화 뒤 GPU 종류·최종 quote·지역/image·CPU/RAM/VRAM·유한 입력/시간·과금 종료·보존/회수를 잠근다. Oracle SSH·작은 archive SCP 왕복은 확인했으며 Pod→Oracle 연결과 사용자 Pod Env는 미설정이다. 카드 VRAM 전체를 항상 가용하다고 가정하지 않는다. |
| RZ-O007 | T1/T2 시계·opening·seed·표본·검정·중단·승격 | 첫 결과 전에 설정한다. 100쌍 smoke, 2,000쌍 final, 95% 하한 > 0은 권고이며 이미 잠긴 프로토콜이 아니다. |
| RZ-O008 | 첫 가중치 이후 모델 구조·반복·warm·정밀도 | 현재 가중치 호환은 C02/C03, 구조 변경은 F03이다. CARD-A05/B01/C02/C03을 모두 채택한 것으로 간주하지 않는다. 각 변경은 독립 대조한다. |
| RZ-O009 | 부분 결정: 최소 의미 계약; 모델별 action/tensor/오차 남음 | 기본 의미·관점·수명은 D025를 따른다. 선택 모델의 policy mapping·history-fill·shape·dtype·오차·deadline 단위는 연결 전에 잠근다. |
| RZ-O010 | 데이터·교사·loss·split·seed·자가대국·훈련 언어 | F01은 계약·누출 검사를 병렬 준비한다. F02 실제 학습 전 provenance·권리·game/opening 분리·목표 관점·비용을 잠근다. |
| RZ-O011 | 부분 결정: 최소 계약 게시; 소비자 인수 남음 | 2026-10-02 초기 작성 당시 Git이 없었으나 후속 원격 추가 요청의 bootstrap은 D028로 진행한다. 실제 Git·브랜치·원격·PR 상태를 착수 때 조회한다. I01은 rz-contracts·최소 workspace·CPU 계약 CI를 게시한다. 각 담당의 소비자 연결과 실제 NN/GPU 실행 성공은 별도 검증한다. |
| RZ-O012 | 규칙 claim·dead-position 범위·runner 판정 | [CONTRACTS](CONTRACTS.md)의 claim/자동 종료·이력·mate 우선순위를 따른다. claim 의사표시·자동 수락·adjudication·지원 범위는 첫 대국 전에 외부 runner와 잠근다. |

## 현재 증거와 후속 기록

초기 문서 작성에서 완료한 것은 참조 조사, 로컬 규약·구현 지시서 작성, 가중치
후보의 공개 자료·표적 메타데이터 조사다. 후속 원격 추가 작업은 실제 커밋·원격 SHA·
PR·검사 조회 결과로 별도 보고한다. 후속 [공통 계약 PR #7](https://github.com/daejunnom/RoveZero/pull/7)에는
최소 Rust 계약과 CPU workspace 검사가 추가됐다. 같은 PR의 commit·환경·실제 검사 결과를
기준으로 증거를 확인하며 계약 crate 자체 성공과 A~F 소비자 인수는 구분한다. 실제 신경망·
목표 GPU 추론·학습·종단 프로파일링·Elo 대국은 초기 계약 게시 단계에서는 수행하지 않았다.
이후 실제 CPU/CUDA·UCI·runner의 source별 인수와 실패·미완료 범위는
[INTEGRATION-STATUS](INTEGRATION-STATUS.md)에 기록한다. 가중치 조사 결과와
실제 권리 확인 범위는 선정 문서에 기록한다.
Accelerate `develop`의 규약과 오래된 로컬 구현 브랜치는 참조 당시 commit·blob으로
[출처 기록](reference/README.md)에 구분했으며 새 RoveZero 구현의 성공 근거로 쓰지 않는다.

후속 결정은 `RZ-D` 또는 `RZ-O` ID를 유지해 상태를 갱신하고 날짜·이유·근거·
대안·검증 범위를 남긴다. 연구 후보의 원문 ID는 바꾸지 않는다. 미확정·기각·불확정
결과를 삭제하거나 문서 채택을 런타임·대국의 성공으로 승격하지 않는다.
