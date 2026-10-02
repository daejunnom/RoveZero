# F01 및 F02 내부 lifecycle 구현 인수 기록

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
- F02 내부 recipe 잠금, 합성 CPU float64 gradient·SGD momentum, frozen/no-update 검사.
- model/optimizer/RNG/sampler/선택/누적 예산 checkpoint·재개, 자체 export·probe 검증.
- MIT 자체 코드·합성 fixture·독립 회귀 검사·사용법.

전체 변경은 `experiments/model-research/`에 있다. Python stdlib만으로 실행한다.
루트 Cargo·CI·`rz-contracts`·A~E 소유 코드는 수정하지 않았다.
패키지/명령·형식·자원 상한은 [README](README.md)에 기록했다.

## 실제 검증

Linux x86_64, Python 3.12.14, CPU 환경에서 수행했다. Rust toolchain과 GPU는 unavailable로
관측했고 CUDA나 학습 프레임워크 설치를 요구하지 않았다.

| 검사 | 명령·입력 | 관측 결과 |
|---|---|---|
| CPU regression | `PYTHONPATH=src python -m unittest discover -s tests -v` | 105 tests 통과 |
| source plan | `rz_data split`, fixture sources, `fixture-demo`, 8:1:1 | source 1개, train 배정, 정상 종료 |
| JSONL audit | `rz_data validate`, fixture manifest/records와 위 plan | 구조 검사 통과, row 1개, 거부 0개, execution_ready=false |
| package/entrypoint | 저장소 밖 source copy에서 `pip install --no-deps --no-build-isolation --no-compile --target ...`, 설치된 `rz-data`/`rz-train` | wheel 생성·설치, training row 6개 감사, source checkpoint 재개와 export 검사 통과 |
| CPU fixture gradient | `rz_training run`, training fixture, seed 17, float64, batch 2, SGD lr 0.1/momentum 0.8, 8 step/16 samples | 실제 weights·출력 변경, validation loss 감소 |
| resume/reproduce | 같은 recipe 8 step 연속 실행 vs 3 step pause+resume | model·optimizer·RNG/sampler·history·best 선택 일치; 시간만 별도 실제 계측 |
| frozen/export | freeze recipe, validation probe import 대조 | weights·optimizer·출력 불변; 자체 export probe 오차 0 |
| 실패/자원 | sampler·validation·checkpoint·export 취소, 수치 실패, 시간 초과·재개, ledger 출력 상한 | 회귀 검사로 재현·typed 결과/복구 상태 확인 |
| 변경 검사 | `git diff --check` | 통과 |
| 독립 요구·코드 검토 | 라벨/이력/관점/실패, split/provenance, 입출력·자원·준비 상태 | 지적한 API 경계·출력 증폭·거부 row 감사 범위 수정 및 회귀 검사 |

fixture source plan digest:
`e215539c2ea6dca03b2637baefbe11ba91eaf5055226bf4597187f7f458620c1`.
fixture audit receipt digest:
`2173394ddeadeed293d9f0ef48eeab0ec0fa9f059edde0e8979e13ef2143c14e`.

training fixture raw records SHA256:
`8f270386a261924b94ef6760147a41275c1501043eb3d4e5f0bfd42568515439`.
seed 17의 8 step 최종 model digest:
`3cc61c850b0230c493bf6a858d5489ea8733c7f491ed0155e0608c947b15f2fd`.
validation weighted loss는 step 1 `2.831684806206051` → step 8
`2.4399378789565906`이었다. 숫자 fixture의 결과이며 실제 체스 품질 지표가 아니다.
`cpu-continuous-v1`와 `cpu-pause-v1`(3 step)→`cpu-resume-v1`의 model/optimizer/
sampler/best/history는 동일했다. `cpu-frozen-v1`은 weights 불변과 step 0 선택,
export validation probe 2개는 float64 출력 오차 0을 확인했다.
소스·검사 commit `8aa798e6fe3e760be125c5c79da869a1a9c46378`의 설치된 CLI도
같은 checkpoint를 재개해 source 연속 실행과 provenance/model/optimizer/sampler/
history/best가 같았다. wheel SHA256은
`2777238990fec741138446b60d882d4f99ea84570e74f9fe126eff67247d09ad`.
이후 인수 기록 수정은 Python 소스를 변경하지 않는다.

검토 중 nested non-finite 값과 결과 context 검사, 긴 group ID의 보고서 증폭,
거부된 row의 provenance/누출 scope 누락, CLI의 I/O 원인·단계 손실을 재현·수정했다. 실패 이력은 WIP 후속
커밋으로 보존한다. 원격 CI는 workflow가 없으며 로컬 검사 통과와 구분한다.

## 미인수 범위와 다음 연결

[PR #7](https://github.com/daejunnom/RoveZero/pull/7)의
`ae7bf5c20c3acdc12ef7e20aa1a88b5853ee99c8`에서 Rust 공통 revision **0.1**의
타입·소유 경계를 확인했다. 확인 당시 open/draft, base `develop`/`9f0bc59`, merge 미실행,
미해결 review thread 0, check run 0이었다. Python wire·Rules/Encoder binding은 제공하지 않는다.
manifest의 `engine_contract_revision`은 null 또는 미검증 선언으로 보존한다.
Rust 계약 발행이나 실제 provider binding을 대신하는 타입을 만들지 않았다.

Rules 상태 복원, 독립 합법 수/종료/이력 대조, 실제 input digest 재계산,
증강의 규칙 의미 대조, teacher label 품질, 실제 데이터 생성·GPU·Maia 학습·paired 대국은
미실행이다. `training_eligible_records=0`이며 F01 전체 인수를 완료했다고 표시하지 않는다.
teacher/source 권리와 외부 pretraining overlap은 선언·미확인 범위를 보고서에 보존한다.

총괄 I01: 게시한 revision 0.1의 F 연결·영속화 경계와 잠정 envelope 호환 검토.
A/A03: 초기 상태+trace 복원, 합법 목록/종료/unknown history 증거 제공.
C: encoder/model identity와 move별 policy/WDL 대조 제공.
E: 실험 manifest·source/split/evaluation digest 연결.
F: 위 provider를 소비하는 adapter와 독립 감사 연결 후 F01 전체 인수 요청.
F02는 C02/C03 실제 추론·CONTROL-0→1·검증 dataset·유한 학습 recipe/예산이 필요하다.
F03 CARD 선택·모델 구조 변경은 아직 미착수다.

F02 내부 lifecycle은 [TRAINING](TRAINING.md)의 stdlib 숫자 fixture 범위다. 실제
Maia 파인튜닝·엔진 추론/변환·GPU·CONTROL 대국 인수로 승격하지 않는다. 전체 PR7 계약을
Python에 재선언하지 않고 향후 F adapter가 연결할 move/order/identity/WDL·수명 요구만 기록했다.
독립 리뷰에서 export metadata, sampler 취소 rollback, export 비용의 시간 초과,
큰 exclusion ledger의 출력 reserve, checkpoint 게시 직후 취소 충돌을 재현·수정했다.

## 산출물 보존

일반적으로 합성 fixture와 검토된 요약을 Git에 넣는다. 원시 output run과 package-check는
저장소 밖 이 작업의 전용 `rovezero-f01`/`rovezero-f02` output root에 두었다. 현재 클라우드 환경
수명 동안 보존하고, 환경 종료 전에 필요한 report/package를 회수한다.
공유 요약에서는 개인 경로·비밀·외부 가중치·raw teacher 기록을 포함하지 않는다.
동일 run ID는 덮어쓰지 않으며 새로운 검사에는 새 run ID를 사용한다.

당시 사용자 지시로 검토한 공개 검증 자료를
[F 보존 커밋](https://github.com/daejunnom/RoveZero/tree/bf38023640b2882e079eafc43e437ea9384b78f9/experiments/model-research/verification/2026-10-03-f-evidence-v1)에 게시했다.
원본 JSON 52개, 과거 commandExecution combined log 22개(실패 2개 포함), 실행 입력·
recipe·split·checkpoint·receipt·export·package text metadata를 보존한다. 별도 보존용
새 실행 13개의 stdout/stderr·명령·소스 SHA·결과, 같은 digest로 재구성한 누락 audit,
receipt.history의 파생 TSV와 누락 원장도 포함한다. 파일별 byte SHA-256은 manifest와
SHA256SUMS에 있다. 원래 execution-time source SHA가 없는 항목과 WIP source 미상은
그대로 표시한다. 개인정보 경로를 마스킹하고 비밀/권리 검사를 거쳤으며 build cache·
바이너리·외부 weights/data는 제외했다. source 구현은 변경하지 않았다. 원격 보존을
확인한 뒤에도 원본 scratch 자료는 삭제하지 않는다.

총괄은 `bf38023640b2882e079eafc43e437ea9384b78f9`의 239개 Git blob·917,150 bytes를
저장소 밖 `reports/coordinator-integration/recovered-pr-evidence/exact-blobs/F-bf38023`에
회수하고 모든 byte identity·길이와 외부 receipt를 확인했다. 독립 감사에서는 manifest
237개, SHA256SUMS 238개, embedded payload digest 83개와 F02 receipt 11개가 맞았다.
연속·pause/resume·installed resume의 model/optimizer/sampler/history/best 정합과
elapsed time·전체 JSON byte 차이를 구분한다. source가 없는 historical 22개(실패 포함),
source `36e619…`의 새 recheck 13개, 재구성 audit·파생 TSV·부재 wheel도 그대로 남긴다.
마스킹 전 클라우드 원본이나 실제 Maia 학습·NN·GPU 인수의 근거로 올리지 않는다.
과거 `bb232606…` remote receipt와 최신 사본의 byte 감사·총괄 회수 receipt는 별개다.
외부 보존은 후속 인수와 별도 정리 지시까지 유지하며, 정확한 archive/receipt SHA와
현재 integration 검사는 [총괄 인수 기록](../../docs/INTEGRATION-STATUS.md)을 따른다.
