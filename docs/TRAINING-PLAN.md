# 데이터·학습 구현 지시서

이 문서는 [v0.2.0 핸드오프](reference/RoveZero_Handoff_v0.2.0_KO.md)의 9~12장,
14장, 17장을 데이터 준비·기존 가중치 미세조정·단일 새 구조 비교의 작업으로 정한다.
연구 경로는 [EXPERIMENTS](EXPERIMENTS.md), 공통 타입은 [CONTRACTS](CONTRACTS.md),
대국·manifest는 [EVALUATION-PROTOCOL](EVALUATION-PROTOCOL.md), 전체 task 배정은
[IMPLEMENTATION-DIRECTIVES](IMPLEMENTATION-DIRECTIVES.md)를 따른다.
이 문서는 구현·인수 지시서이며 실제 학습 완료를 뜻하지 않는다. 현재 F의 CPU 합성
linear fixture lifecycle 범위는 [F TRAINING](../experiments/model-research/TRAINING.md),
실제 모델/추론과 선행 인수는 [INTEGRATION-STATUS](INTEGRATION-STATUS.md)를 따른다.

## 2026-10-10 PALS diagnostic 학습 smoke

이전 PALS 목표는 실제 optimizer 학습을 제외한 구현·인수였다. 그 이력과 zero-step
checkpoint/export 자료는 [PALS 구현 기록](PALS-IMPLEMENTATION.md)에 보존한다.
이번 PR23 후속에서 사용자가 추가한 범위는 **diagnostic fixture를 사용한 작은 nonzero
학습 smoke**다. 본격 학습, 실제 target을 사용한 학습, 제품 learned V의 연결·전략·기력 인수와
PR 병합은 이 범위에 포함하지 않는다. 아래 BT4/F02/F03의 선행 권리·데이터·대조 인수도
이 smoke로 충족한 것으로 처리하지 않는다.

| 구분 | 실제 확인한 범위와 남은 인수 |
|---|---|
| nonzero 진단 비교 | 이전 세 번째 실행에서 P/C/V 각각 연속 4회와 2회+checkpoint/resume+2회를 비교했다. 역할별 8회, 총 24회 actual update에서 모델·AdamW state/step·scheduler·RNG·sampler가 일치했고, 의도한 parameter membership·동결·유한값 검사가 PASS했다. |
| 실패와 누적 비용 | 앞선 실행의 1회와 8회 update 실패 자료를 보존한다. 성공한 24회와 합한 실제 누적 소비는 33회다. 실패·재시도와 checkpoint 자료를 실제 누적 비용 기록에 연결한다. |
| 현재 source 재사용 | 영향 source 8개를 직접 대조하여 이전 diagnostic 결과의 재사용을 확인했다. 이전 실제 실행과 현재 재사용 확인은 별도 근거이며 이번 새 optimizer update는 0회다. noNN/fixture consumer 검사를 새 학습 실행으로 세지 않는다. |
| 실제 target·learned V | 실제 학습 target을 사용한 update는 0회다. diagnostic label/coverage는 실제 수집·권한·unknown mask 검증을 통과한 target coverage와 별도다. V-private tensor의 update/resume 검사는 learned V utility/action의 실제 유용성·기력 근거를 충족하지 않으며 누락 target의 mask를 유지한다. |
| 새 실행 인수 | 현재 실제 CPU/NN/GPU는 자원 대기로 미시작이다. 새 V4/FullLineInteractionV2 baseline의 수치·GPU 수명과 paired pilot은 pending이며 최신 상태는 [후속 인수 기록](PALS-IMPLEMENTATION.md#진단-학습과-남은-실행-인수)을 따른다. 과거 CPU/CI/GPU 실행을 새 기준의 성공으로 승격하지 않는다. |

P/C는 공유 tensor와 해당 role의 private tensor만, V는 private V tensor만 갱신한다.
Repair는 P 경로의 검사에 포함한다. 기존 zero-step
`rz-pals-python-preparation-checkpoint/1` 형식을 유지한다. nonzero
`rz-pals-python-nonzero-checkpoint/1`은 별도 domain이며 실제 AdamW moments/steps·
parameter membership·freeze·scheduler·RNG·sampler와 원래 비용을 복원한다.
invalid load는 검증 전에 model/state를 바꾸지 않는다. CPU FP32,
batch/accumulation 1, AdamW lr 1e-4·betas 0.9/0.999·eps 1e-8·decay 0.01,
clip norm 1·constant schedule의 비교를 사용했다.

구현 접점은 [nonzero trainer](../experiments/model-research/pals/src/rz_pals_model/nonzero_training.py),
[training 계약](../experiments/model-research/pals/src/rz_pals_model/training.py)과
[target coverage](../experiments/model-research/pals/src/rz_pals_model/target_coverage.py)다.

2026-10-10 사용자는 실패와 재시도를 합친 **총 update 한도**를 해제했다. 실행별
CPU 2·15분+정리 30초·출력 2GiB 상한과 actual dispatch/completion·실패·checkpoint
기록은 유지한다. 총 한도 해제는 본격 학습이나 새 유료 자원 사용의 승인이 아니다.
진단 자산은 arena 승격 불가로 유지한다. 후속 계약과 기본값은
[PR23 후속 계약](PALS-FOLLOWUP-CONTRACTS.md), 구현·검사·미실행 구분은
[이번 후속 범위](PALS-IMPLEMENTATION.md#2026-10-10-pr23-후속-범위)를 따른다.

## 0. 2026-10-04 강도용 모델 전환의 우선순위

이 절의 실행·미실행은 2026-10-04 BT4 전환 당시 범위다. 위 PALS diagnostic 후속과
별도 이력으로 보존하며, BT4/F02/F03의 본격 학습 인수를 추가하지 않는다.

사용자는 Maia 강도 한계 조사 후 **BT4-it332 실제 적용·LC0/RoveZero 벤치마크**를
지정했다. [로컬 모델 기록](research/LOCAL-MODEL-BASELINE.md)의 BT4 Rust CPU/CUDA
수치 연결·위치 벤치마크는 실제 실행했고 Maia exact profile은 회귀 기준으로 보존한다.
현재 최소 탐색의 즉시 메이트 선택과 마감/물리 완료를 먼저 대조한다. 강한 weights의
추론 성공과 엔진 강도·학습 인수를 구분한다.

BT4 개별 weights의 학습·수정·배포 권리는 미확인이다. 원본/변환 manifest는
`UNVERIFIED-local-research-only`, `redistribution_ready=false`이며 T1 제작자 허가를
BT4에 전파하지 않는다. 권리가 확인된 T1 구조 fine-tuning과 작은 학생 distillation은
후속 대안으로 보존한다. 현재 F의 `LinearFixture`는 실제 BT4/Maia/T1 trainer나 export
adapter가 아니므로 trainable tensor 복원·frozen round-trip·gradient/optimizer/
checkpoint·ONNX parity와 별도 학습 메모리 인수가 선행한다. BT4의 6 GB 장치 추론
성공을 training 적합성으로 쓰지 않는다. F03과 F02를 섞지 않으며 실제 학습은 미실행이다.
256예제/200 step/20분은 유한 smoke 제안이며 이번 실행값이 아니다.

## 1. 구현 경로와 권한

사용자가 선택한 엔진 기반은 독립 Rust workspace와 자체 MIT 코드다. 외부 코드,
가중치·데이터·opening의 권리는 각각 확인하고 자체 코드의 MIT 표시로 덮지 않는다.
학습 라이브러리·optimizer 실행 스택까지 전부 Rust로 작성할지는 아직 별도 결정이다.
엔진의 언어 결정으로 미지원 autograd·GPU training 기능을 이미 확보했다고 가정하지 않는다.

총괄로 배정된 에이전트가 `rz-contracts`의 최소 공통 계약을 직접 소유한다. 데이터 생성,
trainer·arena는 그 계약을 소비하며 각자가 다른 합법 수/상태/WDL 규칙을 만들지
않는다. TASK-A03은 계약의 적용·검증을 담당한다. 계약 변경은 영향 consumer와
기존 기록의 schema compatibility를 확인해 총괄이 버전으로 발행한다.

F를 포함한 A~F의 클라우드 개발 환경에는 GPU가 없을 수 있다. F01 데이터 계약·
누출 검사와 학습 recipe·checkpoint/resume 경로는 가능한 CPU 또는 mock로 먼저
구현·검증한다. 코드·fixture·CPU smoke의 완료를 실제 teacher 분석·GPU 학습·
파인튜닝 완료로 기록하지 않는다. F02 실제 학습은 권리·비용·유한 예산을 확정한 뒤
지원 환경에서 수행하고 총괄 I02가 실제 추론·학습·평가 근거를 별도로 인수한다.
CUDA가 없으면 예상한 capability 부재와 미실행 범위를 명시하며 GPU 모드를 조용히
CPU 학습으로 바꾸지 않는다. 이 문서 공유는 새 실제 학습이나 유료 GPU 사용을 승인하지 않는다.

| 작업 | 먼저 확보할 입력 | 산출물과 다음 작업의 사용 조건 |
|---|---|---|
| TASK-F01 | 공통 상태/수/평가 타입, 교사·출처·split 설계 | 데이터 생성·복원·라벨/누출 감사 경로와 학습 recipe. mock 기반 I/O는 실제 teacher 데이터와 구별한다. |
| TASK-F02 | TASK-C02 선정 원본 weights·권리/형식, TASK-C03 실제 추론, F01 검증 데이터·유한 학습 예산, CONTROL-0/CONTROL-1 자체 탐색 대조 근거 | 기존 architecture의 fine-tuned weights와 학습 영수증. 같은 CONTROL-1 새 search로 원본 weights와 내부 CONTROL-1→CONTROL-2를 비교한다. |
| TASK-F03 | F02/기준 학습 recipe의 재현 근거, 고정 데이터·search·예산 | architecture 한 가지 변경과 대응 기준 모델, 제거 실험·품질/지연/대국 근거. 다른 후보를 동시에 합치지 않는다. |

F01의 형식·split·감사 구현은 공통 계약 발행 뒤 다른 작업과 병렬 진행할 수 있다.
F02 실제 학습은 weights 권리·형식·원래 인코딩과 실제 reference/GPU 추론,
원본 weights의 baseline PUCT(CONTROL-0)→새 search(CONTROL-1) 대조 근거를 확인한 뒤 시작한다.
첫 새 구조는 F09 관계 입력 또는 A05 잠재 병목 등 **한 가지**를
비교할 제안이며 사용자 선택 전 자동 채택하지 않는다.

`TASK-C02`는 가중치 작업, `CARD-C02`는 공유 recurrent block 연구다.
`TASK-C03`은 실제 NN 추론 작업, `CARD-C03`은 부모 latent warm-start 연구다.
이하 반복 연구에서는 반드시 `CARD-`를 써서 작업 의존성과 혼동하지 않는다.

## 2. 학습 전 잠금과 미결정 처리

다음 값을 train manifest에 잠근다. 미정 값은 설계 기록에 남길 수 있지만 실행
진입점은 `execution_ready=false`로 거부한다. 표의 필드는 기계 스키마 구현 요구이며
현재 유효한 설정·실행 명령을 뜻하지 않는다.

| 묶음 | 잠글 내용 |
|---|---|
| 목적·대조 | 질문·TASK/CARD ID·E/A/S·G/H/C/O, 기준군/변경군, 고정할 search·eval 계약, 성공/보류/중단 기준 |
| 원본 모델 | source·architecture/encoder/head ID, 공개 weights 식별·digest·권리, tensor 이름/shape/dtype·policy mapping·WDL 관점·precision, 실제 inference backend |
| 데이터·교사 | 공개 출처·권리·artifact digest·schema, game/opening/split ID, teacher code/binary/weights/version·search/backend/options·예산·재분석 정책, 알려진 비중복 한계 |
| 학습 recipe | train/validation/holdout digest, loss·가중치, optimizer·schedule·batch·precision·accumulation, checkpoint/selection 규칙, seed 목록·튜닝 탐색 공간 |
| 유한 예산 | teacher/reanalysis/self-play game·position·time, train step/token/sample/time, tuning trial·seed·compile time, CPU/GPU/RAM/VRAM/worker·저장 상한과 취소/자식 종료 |
| 검증 | 독립 규칙 참조·label 감사·leakage 정책, CPU reference/GPU 허용 오차, 품질군·tail risk 기준, fixed-search equal-time 비교와 별도 외부 LC0 확인 |
| 영수증 | 실제 완료 step/sample/시간·최고점 선택 범위·실패·중단·사용 자원, code/config/data/checkpoint digest, checkpoint provenance와 평가 manifest 연결 |

첫 호환 weights·형식·권리 근거는 [WEIGHT-SELECTION](WEIGHT-SELECTION.md)의
Maia1 v1.0이다. RZ-O005/O006/O008/O009/O010의 실제 GPU/학습 예산·backend·
수치 인수·학습 recipe·teacher/data/split는 아직 값이 필요하다. 사용자 요청에 따라
권리와 RTX 4050 Laptop 6GB 자료상 적합성을 조사해 선정했다. 장비 호환 조건은 학습 시간·
VRAM 사용 상한·유료 비용·장시간 실행 예산까지 확정했다는 뜻이 아니다. 문서나
장비 탐지로 유료 GPU·대형 teacher 생성·무제한 학습 권한을 만들지 않는다.

미세조정 전에 현재 weights를 독립 Rust inference에서 실제로 읽고 올바른 출력을
내는지 확인한다. format adapter가 tensor를 읽었다는 것만으로 inference 호환성이
확인된 것은 아니다. 기존 architecture를 유지하는 F02와 입력·body를 바꾸어
기존 weights 일부/전체의 재사용이 불가능해지는 F03을 별도 실험군으로 기록한다.

## 3. TASK-F01 작업 묶음 A: 복원 가능한 데이터와 라벨 계약

**입력:** 공통 PositionState/Move/Eval 계약, rights가 확인된 game·opening·교사.
**출력:** versioned dataset schema, 생성/읽기/감사 경로, provenance ledger.

한 row가 완전 규칙 상태와 모델 입력을 복원할 수 있도록 다음을 저장한다.

| 필드 | 의미와 검증 |
|---|---|
| 정체성 | dataset/schema/record ID, source game/opening ID, 알려진 이력 시작점, state/input/encoding 식별자와 split |
| 상태 복원 | 초기 규칙 상태+합법 이동열 또는 동등한 완전 상태·필요 이력. 미상 FEN 이전 이력을 만들지 않는다. |
| 합법 수 | 표준 move encoding과 promotion, 결정한 순서·목록 hash. label의 move를 단순 index로만 저장하지 않는다. |
| 교사 출처 | teacher identity/version, code/binary·weights digest·권리, search/backend/옵션·장비·예산, label 생성/재분석 시각·상태 |
| policy | 후보별 probability/visits와 정규화/temperature 규칙, bestmove·후보 순위의 출처. 방문 수와 NN raw policy를 구별한다. |
| value | W/D/L·Q 또는 score의 척도/관점/예산·종료 상태. cp를 WDL 또는 확률 Q로 조용히 변환하지 않는다. |
| 실제 결과 | 게임 W/D/L·viewpoint·정확한 종료 이유, engine loss/adjudication/incomplete와 규칙 terminal의 구별 |
| 품질·cost | 유효/실패/부분 label 상태, confidence 출처, teacher 시간·노드/실제 평가·자원, 재분석·선정 이유 |

교사 분석은 유한 탐색의 추정이며 규칙 정답이나 무한 예산의 exact Q가 아니다.
실제 game result와 teacher WDL은 다른 target으로 저장한다. 무승부 확률과
epistemic uncertainty도 구별한다. 미완료·crash·invalid label을 0점 또는 draw
라벨로 채우지 않는다. data filtering의 사유·전후 수·지워진 source ID를 ledger에
남기며 원본 근거를 보존한다.

작은/큰 분석 예산을 짝지어 계산 효용을 연구하면 동일 완전 입력·teacher·옵션을
고정하고 예산만 바꾼 두 label을 연결한다. 미래의 깊은 분석은 supervised target으로
사용할 수 있지만 실제 inference feature에만 존재하는 것처럼 제공하지 않는다.

**완료 근거:** row→상태→독립 합법 수 복원, label/move mapping·viewpoint·유한값
검사, teacher/game result 구별, invalid label 배제가 실제로 동작한다. 작은 읽기
fixture만 통과하면 schema 검사 완료로 기록하고 실제 대형 dataset 검증으로 부풀리지 않는다.
**중단:** 권리·provenance 미확인, 합법 수 대응 오류, 미상 이력 추정, 관점 혼합,
partial/실패 label의 성공 처리 또는 label 생성 비용 추적 불가.

## 4. TASK-F01 작업 묶음 B: split·누출·실패군

**입력:** game/opening provenance와 복원/label 감사가 가능한 dataset.
**출력:** immutable train/validation/holdout split, leakage 보고, failure-stratum ledger.

먼저 game/opening 단위 그룹을 분할하고 나서 row를 만든다. 같은 game의 인접
상태를 무작위 row split로 다른 세트에 흩뜨리지 않는다. opening에서 분기한
game군·같은 seed/교사 재분석 record도 그룹 연결을 고려한다. 정확한 전체 입력
중복·반복 수순·transposition과 가까운 opening 중복을 감사해 그룹 이동 또는
사전 제외 정책을 적용한다. 보드 digest만 같으면 입력이 같은 것도, 이력이 달라
보드만 중복인 record가 항상 무해한 것도 아니다. 목적에 맞는 누출 감사 키를 둔다.

외부 teacher/사전학습 weights의 원래 train corpus를 모르면 최종 holdout과
완전히 비중복이라고 주장하지 않는다. 확인한 source·키·범위와 확인할 수 없는
한계를 기록한다. holdout은 hyperparameter·seed·checkpoint·opening 선택에 쓰지
않고, 여러 번 보고 선택했다면 새 독립 holdout 또는 명시한 선택 편향 보정이 필요하다.

전술 체크·희생·유일 방어·장거리 선 개방·닫힌 구조·endgame·특수 수·긴 누적
갱신·동일 완전 입력의 다른 도달 경로를 failure group으로 유지한다. 모델 변경 후
평균 teacher 일치가 좋아져도 유일 방어 실패·치명적 regret 꼬리가 악화되면 보류한다.
실제 대국 결과를 본 뒤 유리하게 failure group을 재가중하지 않는다.

대칭 증강은 변환 후 폰 방향·차례·castling·EP·카운터·move와 WDL 관점까지
규칙 의미가 보존되는 경우만 허용한다. 임의 board flip·색 교환을 기본 augmentation으로
넣지 않는다. augmentation ID·원본 record 관계를 보존하고 원본과 파생본은 같은
split에 둔다. 독립 Rules 참조로 변환 전후 합법 move 대응을 감사한다.

**완료 근거:** 그룹 split 재현, 정확/근접 중복 정책과 확인 범위, 증강/재분석
source 연결, validation과 holdout 선택 분리가 감사 가능하다.
**중단:** holdout 누출, source ID 제거로 누출 추적 불가, 합법 의미를 깨는 대칭,
새 결과에 따른 유리한 분포 변경.

## 5. TASK-F02: 원본 architecture의 fine-tuning

**입력:** TASK-C02의 권리·형식이 확인된 weights, TASK-C03의 실제 CPU reference·
같은 weights GPU 대조, F01 dataset·split, 원본 weights의 CONTROL-0/CONTROL-1 자체 search
대조 근거와 유한 학습 manifest.
**출력:** baseline train 재현, fine-tuned checkpoint, provenance·비용·품질·inference
호환 영수증, 같은 새 search에서의 원본/미세조정 weights 내부 CONTROL-1→CONTROL-2 대조.

1. 원본 인코더·body·head·move mapping·WDL을 문서화하고 원본 inference fixture를
   확보한다. frozen/no-update recipe의 evaluation mode에서 원본 출력이 유지되는지 확인한다.
2. loss·optimizer·schedule·batch·precision·checkpoint selection을 잠근다.
   처음에는 기본 policy/WDL recipe를 재현하고 추가 loss와 optimizer 교체를 별도 변경으로 둔다.
3. teacher policy/visits와 실제 result·teacher WDL의 사용 비율·viewpoint를 명시한다.
   없는 target을 유효 label처럼 합성하지 않는다. train/validation loss와 calibration,
   policy rank·teacher regret·failure group을 같은 계획으로 수집한다.
4. checkpoint를 내부 evaluator 형식으로 변환하고 tensor/shape/precision·출력 계약을
   대조한다. export/quantization이 값을 바꾸면 A 변경과 별도 cost/오차를 기록한다.
5. 같은 CONTROL-1 새 search·backend·자원·시계·opening에서 원본 weights(CONTROL-1)와
   fine-tuned weights(CONTROL-2)를 비교한다. 원본 weights+baseline PUCT(CONTROL-0)→CONTROL-1의
   결과와 합쳐 원인을 한 숫자로 설명하지 않는다.
6. 실제 LC0 외부 목표의 same-time 대국을 별도 manifest로 확인한다. fine-tuning
   training loss 감소·내부 A/B 개선만으로 LC0 강도 승격을 보고하지 않는다.

자체 model head에 optimizer gradient가 반영되고 실제 weights·출력이 변한 근거가
필요하다. random mock 출력·학습 loop 호출·checkpoint 파일 생성만으로 실제
fine-tuning을 완료했다고 표시하지 않는다. CPU reference 결과, GPU 적합성,
학습 성공, 내부 대국, 외부 LC0 대국을 각각 다른 완료 범위로 기록한다.

**완료 근거:** 변경된 weights를 자체 엔진이 실제로 사용하고, schema/finite/viewpoint·
CPU/GPU 허용 오차·latency·memory·fixed-search 품질 검증이 통과한다.
정해진 예산·seed·선택 조건과 negative/inconclusive 결과도 보존한다.
**중단:** NaN/Inf, output contract 붕괴, loss/label 관점 오류, 지원하지 않는 형식의
조용한 fallback, 유한 예산 초과, holdout 선택 또는 tail failure의 설명 없는 증가.

## 6. TASK-F03: 단일 새 architecture 비교

**입력:** 재현 가능한 recipe·F01 data, TASK-F02의 원본 구조 대조 근거, 고정 search·
output contract·training budget와 선택한 CARD 한 가지.
**출력:** 같은 데이터/예산의 기준 모델과 변경 모델, 동등 조건 추론·품질·ablation 보고.

핸드오프의 첫 신규 모델 제안은 작은 square-token 기준선에 관계 입력(F09/F10)
또는 작은 잠재 병목(A05) 하나를 비교하는 것이다. 실제 선택은 DECISIONS에 잠근다.
기존 weights와 호환되지 않는 입력·architecture는 단순 tokenizer 교체나 같은
weights의 runtime 최적화가 아니다. 호환되지 않는 tensor의 임의 무시·랜덤 채움을
미세조정 성공으로 숨기지 않는다.

첫 비교에서는 dataset/split·output contract·train/튜닝 예산·seed policy·search를
고정하고 input/body/head/loss/optimizer를 한꺼번에 바꾸지 않는다. 구조상 unavoidable
변경은 구성요소·초기화·활성 parameter·연산·메모리 영향과 비교 한계를 기록한다.
동일 step와 동일 wall-time 학습은 다르므로 어느 예산을 기준으로 잠갔는지 명시하고
실제 sample/step/time을 함께 남긴다.

소형 square-token 기준선의 품질/지연과 강한 외부 LC0 기준선의 대국 결과를 구별한다.
품질-지연 Pareto 개선은 후보 선정 근거이고 자동 강도 승격이 아니다. 최종 요소
결합은 성공한 단일 실험만 사용하고 하나씩 제거하는 ablation으로 각 기여를 확인한다.

**완료 근거:** 고정 조건·선택 편향·budget 차이가 명확하고 latency/memory·quality·
실제 equal-time 대국을 각각 재현할 수 있다. 결과가 불확정이면 그대로 남긴다.
**중단:** 여러 변경을 하나의 효과로 주장, unfair training/tuning 예산, weights
비호환 숨김, 외부 baseline을 약화한 비교 또는 failure-tail 악화.

## 7. CARD-C02/C03: 반복·부모 latent의 후속 실험

두 후보는 보존할 핵심 연구지만 runtime baseline 이전 자동 도입 대상이 아니다.
TASK-F03의 후속 단일 실험으로 진입할 때도 아래 순서를 지킨다.

### 7.1 CARD-C02: 고정 반복의 fresh 기준

같은 recurrent block을 고정 횟수 반복하는 fresh 모델을 먼저 둔다. 제안 반복
1/2/4/8은 실제 예산·GPU shape에 맞춰 잠글 후보이며 모든 값을 반드시 실행할
의무는 없다. 반복별 policy/WDL·calibration·tail failure와 실제 시간/VRAM·batch
분포를 측정한다. parameter 공유는 반복 연산이 사라진다는 뜻이 아니다.
같은 wall-time에서 추가 반복과 search 확장의 효용을 비교한다.

정책 증류·WDL·후보 regret·반복별 consistency·추가 계산 효용 head는 후보
loss이며 동시에 기본 적용하지 않는다. loss·head 하나의 변경과 비용을 대조한다.
동적 종료·iteration별 큐·다중 정밀도는 고정 반복 기준 이후 각각 A/S 실험으로 둔다.
추가 계산의 효용은 작은/큰 budget label 관계와 실제 선택 개선으로 검증한다.

### 7.2 CARD-C03: 같은 fresh 모델과 warm 경로 비교

fresh 기준이 안정된 다음 `z0(s')=U(z(s), MoveDelta)`와
`z(t+1)=R(z(t), E(s'))`의 부모 warm-start를 한 변경으로 비교한다.
부모 latent provenance는 parent input·path·model·encoding·precision·budget·
실제 완료 반복을 포함한다. 부모 state 전송·상주량·eviction/restore 비용을 합쳐
계산하고 GPU 상주 여부별로 나눈다. warm 시작 전 latent 생성 비용도 비용 회계에 남긴다.

fresh와 warm을 같은 정확한 child input·고정 budget로 대조하고, 같은 wall-time에서
warm의 계산 절약이 실제 search에 도움이 되는지 따로 확인한다. 같은 보드라도
model input history가 다르면 다른 입력이다. 경로 편향 감사는 **같은 완전 모델 입력**이
다른 ancestor 경로로 나타나는 경우에 실시한다. 경로 독립 목표를 학습해도 warm
state는 근사 namespace에 두며 exact eval cache로 승격하지 않는다.

학습 feature에는 실제 inference 시점에 사용할 수 있는 부모 상태·합법 MoveDelta·child
정확 입력만 넣는다. 미래 PV·깊은 교사 결과·미래 game result는 target이 될 수
있지만 online feature로 주입하지 않는다. cached target·teacher budget 차이가
path consistency의 가짜 성능을 만들지 않는지 감사한다.

failure group에는 체크·포획·승격·희생·긴 선 개방·king/threat delta·특수 수,
long trace·다른 도달 경로·eviction·stale parent를 둔다. 오류 시 같은 자체 모델의
fresh 경로로 복구하고 복구/감사 비용도 온라인 자기 시계에 포함한다. 마감에 복구가
불가능하면 실패 유형을 반환한다. 외부 엔진 호출로 정상 평가를 숨겨 만들지 않는다.

**완료 근거:** fresh/warm·budget/time·path·resident/transfer 효과가 분리되고,
근사 namespace·cancel/deadline·tail risk와 복구 비용이 검증된다.
**중단:** 미상·미래 정보의 online 누출, wrong parent·stale latent 재사용,
정확 cache 오염, 평균만 개선하고 유일 방어 실패 증가, 자기 시간 밖 계산.

## 8. 선택적 학습 기법과 자가대국

optimizer SOAP/AdamW 비교, quantization·low-bit·distillation·LoRA·expert·GRPO·
learned transition·retrieval은 후보 등록부에서 선택한 별도 실험이다. 이름을 나열한
것을 채택·지원·학습 성공으로 기록하지 않는다. LoRA는 바뀐 input/architecture에
원래 weights가 자동 호환되게 하지 않는다. merged adapter의 precision 오차와
동적 adapter가 batch를 나누는 overhead를 검사한다. pruning·quantization의
export/runtime 지원과 실제 shape에서의 속도는 별도 확인한다.

자가대국은 Rules·state/move·NN inference·search·cancel/deadline·dataset label
계약이 안정되고 finite game/position/time·workers·storage가 잠긴 뒤 추가한다.
reward와 실제 결과·teacher value·합법성은 구별한다. 같은 seed/game의 adjacent
rows와 augmentation은 같은 split에 두고, online training을 주 paired 대국에
섞지 않는다. 외부 teacher는 offline data 생성용이며 live 대국 중 호출하면 모든
cost를 포함한 별도 hybrid/composite 실험이다.

## 9. 비용·재현·인수 기록

총 연구 비용은 teacher generation, reanalysis, data read/write·storage,
self-play, train, validation, 여러 seed/trial, compile/kernel tuning과 model export를
구분해 합친다. fastest/best seed 하나의 결과로 구조 일반화를 주장하지 않는다.
budget 내 가능한 여러 seed를 사전 선택하고 best selection의 규칙·trial 총수와
개별 negative/inconclusive 결과를 보존한다. 여러 seed가 불가능하면 그 한계를 적는다.

checkpoint와 report에는 input/model/weights/data/recipe/seed·선택 근거·실행 cost,
output contract 검증 범위·취소/실패·중단 원인과 평가 manifest를 연결한다. 검토된
작은 fixture·재현 설정·요약만 source에 두고 dataset·raw teacher trace·weights·
대형 로그는 저장소 밖 생성물 루트에 보존한다. 개인 절대 경로·비밀은 공유하지 않는다.

F01 인수는 복원/label/split/provenance를 구현·감사한 범위다. F02 인수는 실제
weights 학습·독립 추론 호환·원본/미세조정 대조다. F03 인수는 단일 구조의 동일
데이터/예산 대조다. training loss·puzzle 정답·kernel speed·CPU mock·GPU 실행·
내부 paired 대국·외부 LC0 성과는 각각 다른 증거 상태로 보고한다.
