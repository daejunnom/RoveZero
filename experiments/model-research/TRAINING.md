# F 내부 recipe·CPU fixture 학습·resume·export

`rz-train`은 TASK-F02를 연결하기 위한 **내부 lifecycle 구현**이다. 자체 MIT
합성 숫자 feature와 작은 두 linear head(policy/WDL)로 실제 soft-target cross entropy
gradient와 SGD momentum을 계산한다. stdlib float64 CPU 실행이며 random 출력이나
빈 checkpoint 생성만으로 학습 성공을 대신하지 않는다.

Maia weights·체스 인코더·Rules를 구현하거나 가져오지 않았다. 실제 teacher 데이터,
Maia 원본 구조 fine-tuning, Rust evaluator export 호환, GPU·대국 검증은 미실행이다.
공통 Rust revision 0.1은 PR #7에서 확인했지만 wire가 없으므로 `wire_binding=not_run`,
`execution_ready=false`를 유지한다. 이를 `true`로 바꾸면 실행을 거부한다.

## 입력 잠금과 실행 범위

`fixtures/training/`의 6개 합성 row는 train 3개, validation 2개, holdout 1개다.
source 그룹을 먼저 고정한 split plan과 F01 감사 입력을 포함한다. 숫자 feature는
row의 선언한 full input digest에 연결하지만 실제 C encoder의 출력은 아니다.

`recipe.json`은 아래 값을 잠근다. 자동 튜닝·fallback·schema migration은 없다.

| 묶음 | 현재 지원 |
|---|---|
| scope | TASK-F02 내부 준비, `cpu_fixture`, CPU, float64, worker 1 |
| 데이터 | manifest canonical digest, records/features 원시 파일 SHA256, immutable split plan digest |
| 모델 | fixture adapter ID, feature width, 단일 ordered legal move 배열, seeded initialization |
| target/loss | 명시한 policy 확률·teacher side-to-move [W,D,L], 두 CE loss의 고정 가중치 |
| optimizer | SGD, learning rate·momentum, constant schedule, batch·seed, accumulation 1 |
| 선택 | baseline step 0 포함, validation weighted loss 최소, 동률은 이른 step |
| checkpoint | 저장 간격, model·optimizer·sampler/RNG·누적 step/sample/time·best model·metric ledger |
| 예산 | 총 step/sample/time, record 수, 파일별 입력 bytes, run 전체 출력 bytes |

실행 시 F01 구조·권리 선언·누출 감사를 다시 하고 dataset digest를 대조한다.
입력 변경·미완료 target 합성·cp→WDL 변환·visits 자동 변환은 하지 않는다. 이 adapter는
확률 target의 합 1 오차를 1e-12로 제한한다. F01 저장 형식에서 유효하더라도
policy 또는 teacher WDL이 없거나 fixed move order가 다르면 지원하지 않는 target이다.
`partial/failed/canceled` label은 원래 ID와 사유를 exclusion ledger에 남긴다.
실제 game result는 이 recipe의 target으로 사용하지 않는다.

holdout의 선언·digest·형식은 입력 감사에 포함하지만 forward/loss/selection/probe에는
사용하지 않는다. train과 validation은 모두 비어 있지 않아야 한다. 실제 정식 학습의
calibration·teacher regret·tail failure·독립 품질군 평가 연결은 아직 제공하지 않는다.

## 실행 예

RoveZero checkout의 `experiments/model-research/`에서 실행한다. `ARTIFACT_ROOT`는
저장소 밖의 작업용 artifact 경로다. 같은 run ID는 재사용할 수 없다.

```bash
RZ_F02_OUTPUT_ROOT="${ARTIFACT_ROOT:?저장소 밖 artifact root를 지정하세요}/rovezero-f02"
PYTHONPATH=src python -m rz_training run \
  --recipe fixtures/training/recipe.json \
  --manifest fixtures/training/manifest.json \
  --records fixtures/training/records.jsonl \
  --split-plan fixtures/training/split-plan.json \
  --features fixtures/training/features.json \
  --output-root "$RZ_F02_OUTPUT_ROOT" --run-id cpu-pause-v1 --stop-after 3
PYTHONPATH=src python -m rz_training run \
  --recipe fixtures/training/recipe.json \
  --manifest fixtures/training/manifest.json \
  --records fixtures/training/records.jsonl \
  --split-plan fixtures/training/split-plan.json \
  --features fixtures/training/features.json \
  --output-root "$RZ_F02_OUTPUT_ROOT" --run-id cpu-resume-v1 \
  --resume "$RZ_F02_OUTPUT_ROOT/cpu-pause-v1/resume-checkpoint.json"
PYTHONPATH=src python -m rz_training verify-export \
  --export "$RZ_F02_OUTPUT_ROOT/cpu-resume-v1/export.json" \
  --output-root "$RZ_F02_OUTPUT_ROOT" --run-id cpu-export-verify-v1
```

`--stop-after`는 이번 호출에서 완료할 step 수이며 잠긴 recipe를 바꾸지 않는다.
생략하면 총 예산까지 실행한다. 설치된 `rz-train`도 같은 명령을 제공한다.
`freeze=true`인 별도 recipe는 loss와 batch bookkeeping을 진행하면서 weights와
optimizer를 유지한다. CUDA나 미지원 adapter를 선택하면 CPU로 바꾸지 않고 거부한다.

종료 코드 0은 내부 fixture 실행/계획된 pause/step·sample 예산 완료/자체 export 검사
성공이다. 1은 run 안에서 기록한 실패·취소·시간 예산 종료, 2는 입력·지원 범위·I/O·
출력 상한 오류다. 어느 코드도 실제 F02·엔진 호환·GPU·강도 인수를 뜻하지 않는다.

## 재개·자원·출력 의미

모든 입력 bytes와 작업량에 유한 상한을 둔다. 현재 recipe schema는 최대 10,000 step,
1,000,000 samples, 60초, 10,000 records, 입력 파일별 16 MiB, run 출력 64 MiB,
batch 256, feature width 64, moves 512를 허용한다. fixture는 8 step/16 samples/10초,
파일별 1 MiB/run 1 MiB다. 출력 사전 계산에 checkpoint·metric·exclusion ledger·
validation probe를 포함하며 실행 중 실제 serialized 합계도 검사한다.

시간은 단조 시계로 startup/입력/학습/validation/checkpoint/export를 계측하며 재개 시
checkpoint에 기록한 비용을 더한다. bounded step 사이와 export 후에 확인하는 협력적
제한이므로 진행 중인 한 step·최종 기록의 시간은 초과할 수 있고 영수증에 보존한다.
이는 OS 수준 강제 timeout이나 모든 branching replay의 전역 비용 관리가 아니다.

번호가 있는 checkpoint는 해당 capture까지의 비용을 보존한다. `resume-checkpoint.json`은
export/verification 이후 비용을 담는 최종 재개 지점이며 receipt가 이 파일을 가리킨다.
capture 이후 그 파일을 쓰거나 receipt를 만드는 작은 비용은 receipt의 실제 시간에만
포함된다. 번호가 있는 과거 checkpoint로 재개하면 그 capture 이후의 작업을 다시 수행한다.
step/sample 예산은 재개 때 늘어나지 않는다. 동일 recipe·data·Python 소스 digest를
요구하므로 optimizer 변경, 데이터 교체, 소스 수정 뒤의 checkpoint 재개는 거부한다.
Python runtime이 다른 재현성은 별도 검증이 필요하다.

JSON만 사용하며 pickle을 로드하지 않는다. artifact digest는 무결성·식별용이며 출처
인증 서명은 아니다. 상태·tensor shape·finite·sampler permutation/RNG·sample ledger·
validation 선택을 검사한 뒤 재개한다. 학습 step 도중 취소/수치 오류는 model·optimizer·
sampler·metric을 함께 되돌린다. KeyboardInterrupt는 취소 영수증을 남긴다. I/O 자체
실패는 완전한 checkpoint와 incomplete run을 보존할 수 있고 같은 run에 덮어쓰지 않는다.

출력은 `recipe.json`, 번호별 checkpoint, 최종 `resume-checkpoint.json`, 선택된
`export.json`, `receipt.json`이다. export는 자체 JSON float64 모델이며 별도 변환 없이
validation probe를 import 전후 정확 비교한다. verifier는 source와 metadata 문법,
weights/shape/probe를 확인한다. dataset와 checkpoint 선택의 독립 재감사, Rust f32
변환, tensor/action map·모델 loader 대조는 `not_run`으로 보고한다.

## 계약 연결 때 맞출 항목

Rust 0.1의 Move/Square/promotion과 합법 수 실제 순서, state/input/order identity의
각 의미와 digest 알고리즘, C의 action map/history/model/encoding 식별자를 맞춰야 한다.
F UCI 목록 SHA256를 엔진 order digest로 대체하지 않는다. [W,D,L]은 side-to-move이며
Rust f32 export 시 변환·허용 오차를 별도로 검사한다. runtime handle/epoch/deadline은
checkpoint에서 재사용하지 않는다. terminal/value-only/partial row를 정상 EvalOutput으로
내보내지 않는다. 이 연결은 총괄·A·C의 소유 계약을 소비하는 adapter로 진행한다.
