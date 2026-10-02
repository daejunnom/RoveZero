# E 역할 구현 계획 — TASK-E01/E02/E03

작성일: 2026-10-03, Asia/Seoul. 담당: 사용자가 E 역할로 배정한 이 채팅의 에이전트.
상태: E01 입력 단계의 첫 구현·로컬 검증 완료. 구현 진행과 실제 근거는 9절에 기록한다.

## 1. 계획 수립 당시 기준과 상태

원격 `main`과 `develop`에서 확인한 기준은
`9f0bc598f6b2d8f863fd46af6a4fd73bfef1f0b8`이다. 두 브랜치는 같은 SHA이며,
추적 파일은 규약·설계 문서와 Git 설정이다. Cargo workspace, 실제 공통 Rust 타입,
E 소스, 실행 검사기, CI workflow는 아직 없다.

공개 PR 목록에는 열린 PR 0개, 닫힌 PR 1개가 표시되었다.
[PR #1](https://github.com/daejunnom/RoveZero/pull/1)은 공용 문서 작업이며
`feature/docs-shared-agent-rules`에서 `develop`으로 병합된 상태다.
GitHub API 조회는 `Forbidden`으로 실패했으며 공개 웹 페이지와 Git ref로 위 상태를
확인했다. 최신 SHA의 checks 전체와 리뷰 해결 상태는 확인하지 못했으므로 통과로 보고하지 않는다.

개발 호스트는 Linux x86_64, 노출된 논리 CPU 5개, 메모리 약 17 GiB,
작업 파일시스템 여유 약 30 GiB다. 이는 조회된 호스트 정보이며 실행 예산을 확정한 값이 아니다.
`rustc`, `cargo`, `nvidia-smi`는 PATH에서 발견되지 않았다. Rust 명령은 현재 사용할 수
없으며, GPU 장치/provider 접근은 unknown이다. 목표 RTX 4050 6GB의 검증과 구분한다.

주 기준은 [구현 지시서](../../docs/IMPLEMENTATION-DIRECTIVES.md),
[평가 프로토콜](../../docs/EVALUATION-PROTOCOL.md),
[공통 계약](../../docs/CONTRACTS.md), [실험 계약](../../docs/EXPERIMENTS.md)이다.
[개발 기준](../../docs/ENGINEERING-STANDARDS.md), [협업 규약](../../CONTRIBUTING.md),
[결정 기록](../../docs/DECISIONS.md)과
[v0.2.0 핸드오프](../../docs/reference/RoveZero_Handoff_v0.2.0_KO.md)의 2·17장을 함께 대조했다.
TASK-E01~E03은 구현 담당 작업이며, 핸드오프의 학습 연구 CARD-E01~E12와 다르다.

## 2. 내가 소유할 구현

| 위치 | 구현 책임 |
|---|---|
| `crates/rz-experiments/` | 실행 manifest, 스키마 검증, artifact 식별·digest, 입력 잠금, 실행 영수증·provenance |
| `crates/rz-arena/` | 외부 runner 연결, opening/pair 계획, process 수명·시계 감사, attempt 원장, PGN 감사, 실패 처리·통계 |
| `experiments/baselines/` | 작은 재현 설정·fixture 설명·baseline manifest와 이 계획 |

계획을 baseline 준비 영역에 두어 E의 구현·인수 기준을 같은 소유 범위에서 관리한다.
규약 원문을 바꾸거나 빈 crate를 미리 만드는 작업은 이 계획에 포함하지 않는다.
루트 Cargo·lockfile·CI·공통 설정·실험 목록과 `rz-contracts`는 총괄 소유다.
공통 타입은 게시된 revision을 소비하고, 필요한 추가 항목은 이유·소비자·검사 범위를
정리해 총괄의 변경안으로 제시한다. Arena에 체스 규칙이나 탐색·GPU 스케줄러를 복제하지 않는다.

의존 방향은 `rz-arena → rz-experiments`와 필요한 공통 계약·Position 검증 경계로 한다.
외부 UCI 엔진과 runner는 process adapter로 연결한다. 독립 규칙·통계 참조는 검증용으로
사용하며 제품 규칙의 대체 구현으로 연결하지 않는다. D의 계측은 수동적인 증거 입력으로 소비한다.

## 3. 구현 순서와 완료 조건

| 순서 | TASK와 산출물 | 이 단계를 끝낼 조건 |
|---|---|---|
| 0 | I01 계약 인수, 공통 타입·toolchain·workspace 등록·의존성 제안 | 적용할 revision과 E crate 연결 방법 확인. 계약 전에는 fixture 설계·runner 조사 진행 가능 |
| 1 | E01: manifest 검증·잠금·digest·영수증 | 누락·미지원·변조 입력은 실행 거부, 같은 입력의 재현 식별, 실제 설정 불일치 검출 |
| 2 | E02/E03: 결정적 pair 계획·attempt 원장·재개 | 같은 시작 상태와 흑백 교환 보장, 무효 pair 전체 재시도, 완료 pair 중복 집계 차단 |
| 3 | E02: 기존 runner adapter·process 수명·PGN/clock 감사 | scripted UCI fixture와 CPU 외부 엔진의 정상·실패·취소 대조, 잔여 자식 process 없음 |
| 4 | E03: 실패 분류·WDL/pentanomial·고정 표본 통계·보고 | 독립 손계산/통계 참조 일치, 불완전·경계·불확정 결과 구별, 원시 기록으로 재집계 가능 |
| 5 | E01~E03: RoveZero UCI·D 계측 연결과 인수 패키지 | 동일 SHA/계약/설정의 연결 검증과 미실행 목록 제공. 실제 목표 GPU 정식 대국은 별도 인수 |

실패·ledger 검사를 2단계부터 작성해 runner 결과를 받을 때부터 원본을 보존한다.
4단계의 순수 집계·통계는 3단계의 실제 엔진 준비를 기다리지 않고 fixture로 개발할 수 있다.
각 단계를 작은 의미 단위의 변경으로 나누며 후속 구현은 `develop` 대상 `feature/*` 흐름을 따른다.
원격 공유가 배정된 구현 작업에서는 같은 목표의 draft PR에 WIP·검사·미실행 범위를 기록한다.

## 4. E01 — 실행 명세와 잠금

첫 구현 선택안은 JSON manifest와 JSONL 사건 기록, SHA-256 artifact 식별이다.
실제 의존 라이브러리·버전은 총괄의 workspace 구성에 맞춘다. 다음 이름은 E 내부 설계안이며
이미 게시된 공통 Rust 타입을 의미하지 않는다.

- `ManifestDraft`: 미정 값과 필드별 검증 진단을 보존한다. `execution_ready=false`다.
- `LockedRunManifest`: 필수 입력·버전·단위·예산·정책이 검증된 불변 명세다.
- `LaunchReceipt`: 시작 시 재검증한 binary/weight digest, 적용 옵션·backend·장치의 근거다.
- `RunReceipt`: 실제 실행·중단·실패·산출물 digest를 입력 manifest와 연결한다.
- `ArtifactRef`: 공개 출처, 논리 경로, digest, 권리 근거를 나타낸다. 비밀 파일을 수집하지 않는다.

manifest에는 엔진 source SHA·binary/weight digest·dirty patch, build/toolchain,
model/encoding/search/backend/precision/options, runner·규칙·통계 버전,
장비·자원 한도, opening·history 정책·seed·split, T1/T2/T3와 clock 단위,
draw/timeout/retry 정책, 유한 game/pair/time/worker/output/drain 예산을 넣는다.
통계 방법·CI·표본·중단·승격 기준도 첫 결과 전에 잠근다.

digest 계산용 직렬화의 필드 순서·숫자 표현·버전을 고정하고 digest 자기 필드는 계산에서 제외한다.
미지원 schema, 단위 충돌, placeholder, 빈 digest, 음수/overflow 예산, 불완전 필수 옵션은 거부한다.
동일한 입력 잠금에 실행 후 결과 필드를 덧써서 입력 digest를 바꾸지 않는다.
입력·가중치·시계·통계가 달라지면 새 잠금과 실험군을 사용한다.

요청 옵션과 실제 적용 근거는 따로 기록한다. UCI `readyok`만으로 backend·정밀도·장치나
모든 옵션의 적용을 증명할 수 없으므로, 선택한 엔진 버전의 로그·기본값·관측 수단과 대조한다.
필수 적용 근거가 없으면 정식 비교 준비 완료로 표시하지 않는다. GPU 요구 모드의 CPU fallback도 거부한다.
preflight에서 확인한 설정은 실행 시작 시 다시 대조하며, 평가 opening을 preflight 분석에 사용하지 않는다.

설계할 CLI의 작업은 `validate → lock → plan → run → audit → report`와 `resume`이다.
이는 구현 예정 동작이며 현재 실행 가능한 명령으로 제시하지 않는다.

## 5. E02 — pair 계획과 외부 실행

Opening은 초기 상태와 합법 이동열, 알려진 이력의 시작점·완전성, 규칙/모델 history 정책으로
표현한다. A의 checked transition을 사용하고 외부 독립 참조로 복원·특수 수·카운터를 대조한다.
중간 FEN의 unknown prefix를 과거 이력으로 창작하지 않는다. 처음에는 알려진 전체 이동열을
갖는 fixture를 우선 사용하고 FEN-only 허용 범위는 실행 profile로 명시한다.

한 pair의 두 판은 동일한 완전 P를 복원한다. 후보 백/기준 흑과 기준 백/후보 흑만 교환한다.
실제 차례·보드·기물 색을 바꾸지 않으며 두 판의 시작 상태 digest와 전달 이동열을 대조한다.
pair/game ID, 엔진 seed, 실행 순서와 장비 배정을 결과 전에 결정하고 순서 효과를 균형 있게 배치한다.

기존 Fastchess와 Cute Chess를 첫 비교 대상으로 삼아 하나의 외부 runner를 먼저 연결한다.
선정 기준은 전체 이동열 보존, 색 교대, T1/T2 시계, draw/종료 정책,
불법 수·crash·timeout 구별, PGN/사건 기록, 종료·재시작, Linux/목표 OS 지원과 권리다.
필수 동작을 작은 fixture로 확인한 정확한 버전·binary를 잠그며, 검증 전 특정 runner의 지원을 단정하지 않는다.
외부 구현을 MIT 코드에 복사하지 않고 adapter와 독립 감사기를 작성한다.

runner가 이미 소유한 UCI·시계 기능은 재구현하지 않는다. 관측이 부족한 경우에만 얇은
기록 adapter를 두고 그 overhead를 측정한다. E는 runner 실행 경계와 결과 감사를 소유하며,
B의 RoveZero UCI 구현과 다른 책임으로 유지한다.

UCI fixture는 정상, 준비 지연, malformed/중복/늦은 `bestmove`, 잘못된 승격,
crash, timeout, 출력 과다, 종료 무시를 재현한다. 모든 대기에 deadline을 둔다.
Linux에서는 소유한 process group, Windows에서는 대응 Job Object 등으로 자식 수명을 묶고,
`stop/quit → 유한 grace → 필요한 강제 종료`의 사건·exit/signal·미종료 상태를 보존한다.
한 플랫폼의 성공을 다른 플랫폼에서 실행한 결과로 재사용하지 않는다.

시계는 runner의 monotonic `go` 전달→유효 `bestmove` 수신 경계를 잠근다.
T1의 기본 시간·증분과 T2 movetime, pipe overhead·grace를 별도 필드로 처리한다.
수당 상태 전달 비용도 기록하며 `position` 단계의 숨은 평가를 허용하지 않는다.
warm-up은 평가 opening과 분리하고 이후 reset을 감사한다. Ponder와 상대 차례 계산은 끈다.

단일 GPU 정식 비교는 동시에 한 대국을 기본으로 계획한다. 실제 RAM/VRAM 상한·동시 상주,
`ucinewgame` 초기화와 `bestmove` 뒤 잔여 GPU 작업은 B/D의 사건 및 외부 관측과 대조한다.
외부 runner의 시계만으로 GPU drain이 증명되지는 않는다. 증거가 부족하면
`fairness_unverified`로 남겨 강도 해석을 보류한다.

## 6. E03 — 실패, 재개, 통계

| game attempt 상태 | 처리 |
|---|---|
| `rules_terminal` | Rules의 정확한 종료 원인과 W/D/L 보존 |
| `protocol_adjudicated` | claim 자동 수락·tablebase·기타 사전 판정의 정책 ID·근거와 결과 보존 |
| `engine_loss` | 책임 엔진의 불법 수·crash·timeout을 잠긴 정책대로 패배에 포함하고 빈도 별도 집계 |
| `infrastructure_invalid` | 객관적 장애 근거와 두 판의 attempt 보존, pair 전체를 새 attempt로 제한 재시도 |
| `incomplete` | 예산 종료·취소·미완료를 별도 보고. 무승부 점수를 넣지 않음 |
| `contract_invalid` | 상태·모델·시계·자원·manifest 불일치의 영향 범위를 남기고 해석 중단 |

pair ID와 pair/game attempt ID를 분리한다. 모든 재시도·resume 사건과 새 process를 기록한다.
완료 pair만 한 번 집계하고, 무효 판의 상대 판만 골라 살리거나 엔진 패배를 재실행하지 않는다.
부분 파일·중복 사건·중단 중 원장 쓰기를 탐지할 수 있도록 사건 순번과 완결 표시를 두고,
재개 시 잠긴 입력·artifact·원장 일관성을 먼저 검사한다. 잘린 기록을 성공으로 복원하지 않는다.
양쪽 장애·동시 timeout·최대 수·부분 pair 재개 정책도 사전 설정에 포함한다.
안전하게 이어갈 수 없는 미완료 pair는 그대로 남기고, 사전 허용된 pair 단위 재시도만 수행한다.

Rules의 checkmate/stalemate, claim 가능 3회 반복/50수, 자동 5회 반복/75수와
마지막 수의 mate 우선순위를 runner profile과 대조한다. claim 가능 상태를 자동 terminal로
바꾸지 않으며, dead-position 검사의 지원 범위를 명시한다.

집계는 후보 관점 W/D/L, 실제 완료 pair 수 M, 합계 점수 0/0.5/1/1.5/2의 빈도
`n0..n4`, 실패·제외·attempt 수를 함께 낸다.
`s=(W+0.5D)/(W+D+L)`과 `s=(0n0+0.5n1+n2+1.5n3+2n4)/(2M)`가 일치해야 한다.
전체 attempt 보고와 완료 pair 통계의 분모를 분리하며, 1점 pair의 DD와 WL도 원시 WDL에 남긴다.
빈 표본은 unavailable, s=0/1의 logistic Elo는 비유한 경계 상태로 표현하고 유한 숫자를 만들지 않는다.

첫 통계 구현은 최종 holdout용 fixed-sample 경로를 우선한다. CI의 도입 제안은
opening을 군집 단위로 재표집하면서 내부 pair를 함께 보존하는 cluster bootstrap이다.
반복 opening을 독립 표본으로 세지 않고, 반복 수·seed·CI 수준·구간 방식·최소 유효 군집
조건을 잠근다. 적은 표본·퇴화 분산·경계 표본에서 보장되지 않는 CI를 확정값으로 내지 않으며,
독립 통계 참조로 검증한 지원 범위 밖이면 불확정과 사유를 보고한다.
이 방법은 구현 제안이며 정식 실험의 통계 계획은 첫 결과 전에 별도로 확정한다.

개발용 순차 검정은 검증된 paired SPRT/GSPRT 구현을 별도 adapter로 연결하는 후속 단계다.
구현·버전·H0/H1·alpha/beta·Elo 척도·예산·경계를 잠그고 독립 참조 대조를 통과한 모드만 제공한다.
미지원 순차 설정은 명시 거부한다. 순차 종료 표본에 fixed-sample CI를 붙이지 않는다.
일반 logistic Elo와 normalized Elo, 다른 장비·시간·opening 분포의 결과를 합치지 않는다.
예산 종료나 필수 증거 부족은 `inconclusive`이며 자동 승격시키지 않는다.

## 7. 의미 있는 검증과 인수

| 검증 묶음 | 핵심 입력과 확인할 실패 |
|---|---|
| 명세·잠금 | 누락 필드·빈/틀린 digest·변조 binary·미지원 schema·단위/예산 overflow·옵션 미적용 거부 |
| 상태·pair | 흑 차례 시작, 캐슬링·EP·네 승격·카운터·known/unknown history, 잘못된 색 교환과 이동열 손실 검출 |
| UCI·수명 | 준비/착수/종료 timeout, duplicate/late 출력, crash·출력 한도·취소 시 자식 종료 |
| 원장·재개 | pair 절반 완료, infra 재시도, 중복 import, 잘린 사건, digest 변경 resume, 엔진 패배 재분류 차단 |
| 규칙·정책 | claim/자동 종료 구별, 75수 mate 우선, 미지원 dead position·최대 수 도달을 가짜 무승부로 처리하지 않음 |
| 통계 | 손계산 pair와 독립 참조, 반복 opening 군집, 0표본·모두 승/패·DD/WL 구별, 조건 혼합 거부 |
| 공정성 | clock 차감·증분, 상대 차례 계산·늦은 GPU 완료·reset 누락·메모리 초과의 해석 중단 |

예를 들어 pair 결과 LL, DL, DD, WL, WD, WW는 W=D=L=4,
`n=[1,1,2,1,1]`, M=6, s=0.5다. 이런 독립 손계산을 집계의 기준으로 사용하며,
구현 출력을 다시 기대값으로 사용하는 검사로 대체하지 않는다.

CPU/fixture 인수는 manifest→계획→runner 사건→원장→감사→재집계까지 재현되어야 한다.
실제 CPU 외부 엔진 smoke는 별도 identity와 유한 예산으로 실행하고 RoveZero/신경망 결과와 구분한다.
총괄의 실제 toolchain·workspace 게시 후 E crate의 format/lint/unit/integration 검사를 연결한다.
현재 없는 Cargo·검사 명령을 실행한 것으로 기록하지 않는다.

인계물에는 TASK·기준/구현 SHA·계약 revision·실제 명령·feature·fixture digest·seed·자원 상한,
결과·실패·미실행 목록과 후속 검증자를 넣는다. GPU 정식 대국은 C03/G2, B 시간 제어,
D의 잔여 작업·자원 증거와 잠긴 LC0/runner/통계 계획을 갖춘 지원 환경에서 I02가 별도로 인수한다.

## 8. 다른 역할에서 받아야 할 것과 실행 전 미결정

| 담당 | E가 소비할 결과 | 필요한 시점 |
|---|---|---|
| 총괄 I | 최소 계약 revision, toolchain/workspace/CI 연결, 공통 오류·예산·ID 의미 | crate 연결과 CPU/fixture 인수 |
| A | 상태 복원·합법 수·종료/claim·history completeness API와 독립 검증 근거 | 실제 opening/PGN 감사 |
| B | 실행 가능한 UCI binary, 옵션·clock·stop/newgame 동작과 search ID | RoveZero 연결 검사 |
| C | model/encoding/weight/backend/precision identity와 실제 추론 인수 근거 | 실제 NN 비교 |
| D | passive 사건·자원·deadline/drain/reset 증거와 계측 overhead | 정식 시간·GPU 공정성 감사 |
| F | tuning/holdout opening 분리·누출 검사와 학습 가중치 provenance | 학습 효과 및 최종 holdout 비교 |

대기 중에도 manifest 순수 검증, pair/ledger 설계, 통계 fixture와 외부 runner 적합성
조사는 진행할 수 있다. 공통 타입을 임의 복제하거나 모든 A~D 구현 완료를 선행 조건으로 만들지 않는다.

정식 실행 전 잠금 대상은 RZ-O005의 LC0 identity, O006의 실제 자원·예산,
O007의 opening·시계·seed·표본·검정·중단·승격과 O012의 claim/판정 profile이다.
문서의 100쌍 smoke, 2,000쌍 final, T1 60+0.6/180+1.8초,
T2 100/1,000ms는 제안이며 이 계획으로 실행값을 확정하지 않는다.
내부 W0는 선정된 Maia1 v1.0을 사용하되, 같은 Maia의 LC0 연결 검증을 강한 LC0 대비
목표 달성으로 해석하지 않는다. CONTROL-0→1, CONTROL-1→2, D03 runtime A/B와
외부 LC0 비교는 서로 다른 원인을 검증하는 실험군으로 유지한다.

원시 PGN·JSONL·보고서·binary·weights는 저장소 밖 실행별 artifact root에 둔다.
실행 전 `output_root`의 소유·보존 기간·작업 종료 전 회수 위치를 정하고,
공유 manifest에는 `${ARTIFACT_ROOT}` 기준 논리 경로만 남긴다.
저장소에는 작은 독립 fixture·재현 설정·검토 가능한 요약만 포함한다.

첫 실제 구현 단위는 E01의 draft/locked manifest와 필수 필드·digest·예산 검증이다.
초기 계획 작성 시에는 엔진 코드·가중치 다운로드·대국·CI 실행과 원격 변경을 수행하지 않았다.

## 9. 첫 구현 진행 — 2026-10-03, Asia/Seoul

`crates/rz-experiments`에 E01의 typed manifest, strict JSON 검증, canonical SHA-256
입력 잠금과 bounded artifact 대조를 구현했다. `validate`, `lock`, `verify` CLI로
합성 입력의 잠금·재검증이 가능하다. 실제 실행 시작·옵션 관측·결과 영수증은 아직
구현하지 않았으며 모든 잠금은 `execution_ready=false`다. 상세 계약·명령·한계는
[crate README](../../crates/rz-experiments/README.md)에 기록한다.

계획 수립 이후 Rust toolchain을 설치했다. Linux x86_64에서 Rust 1.99.0으로
CLI 9개와 library 22개, 총 31개 테스트 및 fmt/clippy 검사를 통과했고,
선언한 MSRV Rust 1.85.0에서도 `cargo check --locked`를 통과했다.
독립 Python canonical JSON 계산과 잠금 digest가 일치하며, 합성 artifact의 크기와
SHA-256을 실제 파일에서 대조했다. 의존성 resolution도 별도 lockfile로 보존한다.
이는 입력 단계의 CPU 검사이며 CI·실제 엔진 대국·NN 수치·GPU·통계 계산 근거가 아니다.

루트 workspace·Cargo.lock·CI·공통 타입은 총괄 I01 소유로 남겨 두었다.
다음 구현 단위는 E02의 동일한 전체 시작 상태를 공유하는 color-swapped pair 계획과
attempt/failure 원장이다. A의 상태 계약과 독립 opening 감사, runner의 실제 옵션·시계·
종료 사건 연결 후 실행 인수를 진행한다. E03 집계·군집 통계·독립 holdout 비교는 후속이다.
