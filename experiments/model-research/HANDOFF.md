# TASK-F01 구현 인수 기록

기록일: 2026-10-03 (Asia/Seoul). 담당: F.
기준 `develop`: `9f0bc598f6b2d8f863fd46af6a4fd73bfef1f0b8`.
작업 branch: `feature/f01-data-audit`, [draft PR #2](https://github.com/daejunnom/RoveZero/pull/2).
최종 head SHA와 해당 commit의 검사 결과는 PR 본문에 기록한다.

## 제공 범위

- F 소유의 잠정 데이터 envelope v1, strict JSON과 deterministic SHA-256.
- 라벨의 명시적 move 대응·승격·관점·단위·유한값·정규화·cost budget 검사.
- source game/opening/lineage/seed 연결 성분을 먼저 나누는 immutable split plan.
- 정확 input identity 중복, source 그룹·파생 parent·cycle, board/state overlap 위험 감사.
- bounded JSONL CLI와 거부/제외 provenance ledger, source 밖 immutable run 출력.
- MIT 자체 코드·합성 fixture·독립 회귀 검사·사용법.

전체 변경은 `experiments/model-research/`에 있다. Python stdlib만으로 실행한다.
루트 Cargo·CI·`rz-contracts`·A~E 소유 코드는 수정하지 않았다.
패키지/명령·형식·자원 상한은 [README](README.md)에 기록했다.

## 실제 검증

Linux x86_64, Python 3.12.14, CPU 환경에서 수행했다. Rust toolchain과 GPU는 unavailable로
관측했고 CUDA나 학습 프레임워크 설치를 요구하지 않았다.

| 검사 | 명령·입력 | 관측 결과 |
|---|---|---|
| CPU regression | `PYTHONPATH=src python -m unittest discover -s tests -v` | 58 tests 통과 |
| source plan | `rz_data split`, fixture sources, `fixture-demo`, 8:1:1 | source 1개, train 배정, 정상 종료 |
| JSONL audit | `rz_data validate`, fixture manifest/records와 위 plan | 구조 검사 통과, row 1개, 거부 0개, execution_ready=false |
| package/entrypoint | 저장소 밖 source copy에서 `pip install --no-deps --no-build-isolation --no-compile --target ...`, 설치된 `rz-data --help` 및 fixture validate | wheel 생성·설치·CLI 실행 통과 |
| 변경 검사 | `git diff --check` | 통과 |
| 독립 요구·코드 검토 | 라벨/이력/관점/실패, split/provenance, 입출력·자원·준비 상태 | 지적한 API 경계·출력 증폭·거부 row 감사 범위 수정 및 회귀 검사 |

fixture source plan digest:
`e215539c2ea6dca03b2637baefbe11ba91eaf5055226bf4597187f7f458620c1`.
fixture audit receipt digest:
`2173394ddeadeed293d9f0ef48eeab0ec0fa9f059edde0e8979e13ef2143c14e`.

검토 중 nested non-finite 값과 결과 context 검사, 긴 group ID의 보고서 증폭,
거부된 row의 provenance/누출 scope 누락, CLI의 I/O 원인·단계 손실을 재현·수정했다. 실패 이력은 WIP 후속
커밋으로 보존한다. 원격 CI는 workflow가 없으며 로컬 검사 통과와 구분한다.

## 미인수 범위와 다음 연결

공통 Rust 계약은 문서 초안이며 실제 revision·wire type이 게시되지 않았다.
manifest의 `engine_contract_revision`은 null 또는 미검증 선언으로 보존한다.
Rust 계약 발행이나 실제 provider binding을 대신하는 타입을 만들지 않았다.

Rules 상태 복원, 독립 합법 수/종료/이력 대조, 실제 input digest 재계산,
증강의 규칙 의미 대조, teacher label 품질, 실제 데이터 생성·GPU·학습·paired 대국은
미실행이다. `training_eligible_records=0`이며 F01 전체 인수를 완료했다고 표시하지 않는다.
teacher/source 권리와 외부 pretraining overlap은 선언·미확인 범위를 보고서에 보존한다.

총괄 I01: 공통 revision·영속화 경계 게시와 이 잠정 envelope의 호환 검토.
A/A03: 초기 상태+trace 복원, 합법 목록/종료/unknown history 증거 제공.
C: encoder/model identity와 move별 policy/WDL 대조 제공.
E: 실험 manifest·source/split/evaluation digest 연결.
F: 위 provider를 소비하는 adapter와 독립 감사 연결 후 F01 전체 인수 요청.
F02는 C02/C03 실제 추론·CONTROL-0→1·검증 dataset·유한 학습 recipe/예산이 필요하다.
F03 CARD 선택·모델 구조 변경은 아직 미착수다.

## 산출물 보존

합성 fixture와 검토된 요약만 Git에 넣는다. 원시 output run과 package-check는
저장소 밖 이 작업의 전용 `rovezero-f01` output root에 두었다. 현재 클라우드 환경
수명 동안 보존하고, 환경 종료 전에 필요한 report/package를 회수한다.
공유 요약에서는 개인 경로·비밀·외부 가중치·raw teacher 기록을 포함하지 않는다.
동일 run ID는 덮어쓰지 않으며 새로운 검사에는 새 run ID를 사용한다.
