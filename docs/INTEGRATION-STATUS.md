# 총괄 통합 인수 상태

기준일: 2026-10-03, Asia/Seoul. TASK-I01/I02의 source 관찰·수동 연결·실제 검사를
구분하는 기록이다. 상세 적용은 [CONTRACT-ADOPTION](CONTRACT-ADOPTION.md),
작업 소유는 [IMPLEMENTATION-DIRECTIVES](IMPLEMENTATION-DIRECTIVES.md)를 따른다.
문서 작성에서 Git 작업·검사·학습·대국을 수행하지 않았다. 실제 명령·결과·최종 SHA는
총괄이 실행 후 채운다. 원시 근거는 저장소 밖에 보존하고 논리 경로·digest만 연결한다.

## 기준 조합

공통 계약 [PR #7](https://github.com/daejunnom/RoveZero/pull/7)은 병합됐다.
현재 총괄 develop base는 `3e3cd80533a69ee2118f1c130e72a1cb0745ae8c`,
통합은 [Draft PR #9](https://github.com/daejunnom/RoveZero/pull/9)다.
다음은 총괄이 포착한 담당 source pin이며 최종 integration SHA가 아니다.
A~D는 총괄 제공 prefix를 표시하고 실제 인수 영수증에는 full SHA를 기록한다.

| 담당 | PR | captured source head |
|---|---|---|
| A | #3 | `118dc03` |
| B | #6 | `e80be2f` |
| C | #5 | `0a56367` |
| D | #4 | `83c2869` |
| E | #8 | `383eb05c55f65cd2df5c8642a7f374e50be084b5` |
| F | #2 | `36e619bae238a587e04c0a4bae24624eb5414aec` |

## TASK-I01: workspace와 단일 계약

- [x] Root Cargo에 10개 member 선언: `rz-contracts`, `rz-position`, `rz-search`,
  `rz-uci`, `rz-encoding`, `rz-eval`, `rz-runtime`, `rz-telemetry`, `rz-experiments`, `rz-arena`.
- [x] Root path 의존에서 `rz-contracts=0.1.0`을 단일 공통 계약으로 지정.
- [x] 선언 확인: toolchain `1.96.0`, workspace rust-version `1.90`; common crate
  MSRV `1.85`는 별도 범위이며 전체 workspace의 1.85 지원을 뜻하지 않는다.
- [ ] 모든 consumer의 실제 path·revision·nested workspace 제거·feature·오류·수명
  adapter 정합: 총괄 수동 연결 진행 중, 동일 integration SHA 인수는 pending.
- [ ] 실제 Cargo resolution·format/lint/build/test·CPU 기본 CI 결과: pending.

## TASK-I02: 단계별 증거

| 단계 | 현재 확인한 범위 | 별도 인수와 상태 |
|---|---|---|
| Source 반입·adapter | 위 captured pins와 root member/path 선언; 총괄 수동 정합 진행 | adapter diff·최종 integration SHA·실제 소비자 검사 pending |
| CPU/mock binary·trace | 각 담당의 독립 코드·fixture/test 정의 | root 같은 SHA의 실제 UCI binary·연결 trace·cancel/drain·전체 CPU 검사 pending |
| 실제 CPU NN | C의 모델 입력/loader/provider 준비와 수치 접점 | 선택 weights의 실제 독립 reference parity·engine 연결 pending |
| 목표 GPU | 목표 RTX 4050 6GB와 준비 provider를 개발 호스트와 구분 | 실제 장치·precision·batch·수치·buffer·메모리·D02 종단 계측 pending |
| 실제 F02 학습 | F CPU 숫자 fixture gradient·checkpoint/export/resume 제공 | Maia weights·실제 encoder·검증 data·유한 예산·CONTROL-1→2 인수 pending |
| 정식 paired 대국 | E manifest·pair planner·attempt ledger·원시 WDL/n0..n4 제공 | 실제 runner/PGN Rules·시계/자원 공정성·NN·군집 통계·holdout 인수 pending |

E/F 읽기 전용 source 리뷰에서 E test 선언 74개(Unix 조건부 4개), F test method
105개를 확인했다. 이는 실행 결과가 아니다. E의 structural_only/readiness=false,
F의 readiness=false·training_eligible_records=0·wire/Rules/encoder not_run을 유지한다.
단독 package의 과거 검사 보고·source 선언 수·root 통합 검사를 합산하지 않는다.

## 실제 검사 영수증 — pending

아래 명령은 CPU 기본 범위의 계획이며 이 기록에서 실행하지 않았다. source·feature·
입력·환경과 실행 가능성을 총괄이 확인하고 실제 사용한 명령을 확정한다.

| 계획 명령/검사 | 상태 |
|---|---|
| `cargo +1.96.0 fmt --all -- --check` | pending |
| `cargo +1.96.0 clippy --workspace --all-targets --locked -- -D warnings` | pending |
| `cargo +1.96.0 test --workspace --all-targets --locked` | pending |
| `cargo +1.96.0 build -p rz-uci --locked`와 실제 bounded UCI/mock trace | pending; trace 명령·fixture 총괄 확정 |
| F Python unittest, import path=`experiments/model-research/src` | pending; 실제 Python·명령·출력 루트 총괄 확정 |
| CPU NN·목표 GPU·실제 학습·정식 arena | pending; 각각 지원 환경·입력·유한 예산을 별도로 잠금 |

각 실행 뒤 총괄은 integration SHA, 담당 full source heads, 계약 revision, 명령·OS·
toolchain·backend/feature·fixture/weights/config digest·자원 한도, 실제 결과·실패·
skip/미실행, 외부 산출물 참조·digest와 다음 검증자를 기록한다. CI 요청·관측 완료·
재사용 확인도 분리한다. 최종 integration SHA와 전체 결과는 현재 pending이다.
