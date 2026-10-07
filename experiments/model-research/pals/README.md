# PALS 실제 신경망 전방 계산·초기화·export

이 격리 패키지는 Rust 제품 엔진에 연결할 PALS 모델의 실제 전방 계산을 정의합니다.
기존 `rz_training` 합성 선형 fixture와 별개이며 학습·optimizer step 명령은 없습니다.
`init`은 seed에 따른 **미학습 무작위 가중치**만 만들고 `trained=false`, `training_steps=0`,
외부 가중치·교사 미사용을 기록합니다. 실제 학습 및 기력 인수를 주장하지 않습니다.

고정 구성은 폭 384, Q/KV head 6/2, head dimension 64, 보드 64칸+metadata 2토큰의
2 block encoder, record 내부 4개 구조 필드 토큰의 독립 1 block encoder,
private latent 16×384, 공유 reader 2 block의 2회 반복, 역할별 SwiGLU 1024입니다.
공개 memory K/V는 역할 중립적인 단일 projection bank이고 latent self-attention은
반복마다 재계산합니다. P/C/V는 작업자가 선택하며 role routing softmax는 없습니다.
128개 record·256개 합법 후보·128개 이탈 지점 상한을 넘으면 오류입니다.
Rust의 `PalsModelInput`은 required critical record 누락도 거부합니다.

승격은 공통 `Move16`과 같은 `0=none, 1=queen, 2=rook, 3=bishop, 4=knight`입니다.
기물은 a1..h8 순서로 `0=empty`, `1..6=white P/N/B/R/Q/K`, `7..12=black`입니다.
WDL은 입력 포지션의 차례 관점입니다. 실제 합법성·이력·종료는 Rules가 소유하며,
모델의 숫자 fixture는 합법 체스·학습 데이터로 사용하지 않습니다.

Python 3.11~3.13, torch 2.8.0, numpy 2.2.6, onnx 1.19.0을 고정합니다.
CPU 설치에서는 torch wheel을 공식 CPU index에서 먼저 설치해 CUDA 수 GB 의존성을
추가하지 않습니다. 수치 참조는 별도 optional `onnxruntime==1.22.0`입니다.
이는 기존 Rust `ort=2.0.0-rc.10`의 native ORT 1.22.0과 같은 버전이며 실제 Rust/GPU
인수가 끝나기 전에는 버전 일치만으로 호환 통과를 표시하지 않습니다.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e './experiments/model-research/pals[reference]'
python -m unittest discover -s experiments/model-research/pals/tests -v
python -m rz_pals_model.cli flops --role critic --records 128 --candidates 32 --divergences 16
python -m rz_pals_model.cli init --seed 1 --output "$OUTPUT_ROOT/initialization"
python -m rz_pals_model.cli export --checkpoint "$OUTPUT_ROOT/initialization/untrained.pt" --output "$OUTPUT_ROOT/export-pc"
python -m rz_pals_model.cli numeric-check --checkpoint "$OUTPUT_ROOT/initialization/untrained.pt" --export "$OUTPUT_ROOT/export-pc"
python -m rz_pals_model.cli rust-fixtures --checkpoint "$OUTPUT_ROOT/initialization/untrained.pt" --export "$OUTPUT_ROOT/export-pc" --output "$OUTPUT_ROOT/export-pc/rust-fixtures.json"
```

`OUTPUT_ROOT`는 저장소 밖의 작업 소유 경로여야 합니다. 모델·원시 출력은 Git에 넣지
않으며 기존 결과 덮어쓰기·불완전 파일에 대한 자동 재시도는 거부합니다.
기본 export는 공개 encoder와 P/C graph만 포함하고 V private weight·module을 제거합니다.
`--include-validator`는 학습용 전방 검사 artifact에만 사용합니다. `export.json`은 graph
SHA-256, 실제 graph 입력·출력 이름/shape/dtype, opset 17, precision, provenance를 보존합니다.
P/C graph의 사용하지 않는 V·C 입력은 ONNX export에서 제거될 수 있으므로 graph에
실제로 존재하는 이름을 기준으로 바인딩해야 합니다.

후속 제품용 `--layout shared_pc_if`는 공개 encoder와 공통 P/C graph의 **두 session**
형식입니다. 공통 reader·후보 임베딩의 각 parameter는 graph 바깥 initializer 한 벌에
놓고, 역할별 초기 latent·네 FFN 호출·최종 head만 ONNX `If` 여섯 개로 hard route합니다.
두 역할을 모두 실행한 뒤 `Where`로 고르는 구현이 아닙니다. batch 전체에는 하나의
scalar BOOL 역할만 지정하며 V는 포함하지 않습니다. P의 divergence tensor는 shape만
맞춘 0이고 의미 출력에서는 제거합니다. 실제 학습용 Torch 모듈·parameter 이름은
기존과 같아 export 형식 변경이 학습 구조를 바꾸지 않습니다.

```bash
# native CPU-only pals_rules_descriptor의 stdout을 저장한 선언을 명시합니다.
python -m rz_pals_model.cli export --checkpoint "$OUTPUT_ROOT/initialization/untrained.pt" --output "$OUTPUT_ROOT/export-shared" --layout shared_pc_if --rules-profile-json "$OUTPUT_ROOT/rules-profile.json"
python -m rz_pals_model.cli numeric-check --checkpoint "$OUTPUT_ROOT/initialization/untrained.pt" --export "$OUTPUT_ROOT/export-shared"
```

새 artifact schema `rovezero.pals-model.v2`와 `layout_revision=1`은 모델 의미 v1과
구별됩니다. 기존 v1 separate P/C·P/C/V manifest는 자동 변환 없이 계속 읽습니다.
새 형식은 실제 Rules 입력 필드의 순서·정규화·누락 의미를 설명하는 선언이 필수이며,
58개 의미 문자열을 `u64 little-endian UTF-8 byte length + UTF-8 bytes` 순서로 SHA-256
검증합니다. source SHA와 선언 JSON의 raw·canonical SHA도 별도로 보존합니다.
이 pin은 입력 의미의 일치 선언이며 미학습 가중치의 적합성·기력을 증명하지 않습니다.

`reader_initializer_bank`는 공유 parameter의 이름·byte 수·SHA와 분기 내부 복제 부재를
기록합니다. 이는 ONNX 직렬화 단일 소유의 증거입니다. ORT prepack 복제와 실제 VRAM
공유는 `unknown`이며 native 검사로 따로 인수합니다. CPU 수치 검사에는 최적화를 끈
ORT profile을 사용해 선택된 private branch만 실행됐는지 확인하는 witness가 포함됩니다.
선택·비활성 node 목록과 profile SHA를 보고하며 임시 원시 profile은 회수·삭제합니다.
이 witness를 GPU·제품 최적화 경로의 성능 또는 물리 수명 증거로 승격하지 않습니다.
Rust fixture는 public K/V·mask의 독립 Torch 수치도 함께 전달합니다.

`training.py`는 실제 학습을 시작하지 않고 데이터·loss·AdamW·재개 상태를 준비하는
라이브러리입니다. `load_collected_dataset`은 외부에서 따로 등록한 collector receipt SHA와
encoder source SHA를 받아 `records.jsonl`, `native-inputs.jsonl`, `source-registry.jsonl`,
`split.jsonl`의 식별을 검사합니다. 실제 own CPU 수집 기록의 `rz-pals-data/2` 입력 seal과
native tensor sidecar를 함께 요구합니다. FEN·관측 hash만으로 metadata·이력·public record
특징을 임의 복원하지 않으며 target을 encoder 입력에 섞지 않습니다. sidecar의 원래 tensor
JSON byte SHA, encoding·history·model epoch, record 순서·revision·critical 표시와 합법
후보 순서를 확인한 다음 같은 입력에 연결합니다.

collator는 record·candidate의 padding mask와 policy·WDL·C divergence·V task의 target
mask를 각각 반환합니다. 값이 미완료·미관측이면 WDL target을 mask하며 임의 0점·무승부를
정답으로 넣지 않습니다. C 이탈·V 작업에는 해당 line/ply 및 branch/profile/budget의
명시적 context가 필요합니다. 현재 native role-query sidecar만으로 이를 만들어 내지
않습니다. `masked_losses`는 활성 후보만 policy 정규화에 넣고, 역할별로 유효한 target만
loss에 포함합니다. 모든 target이 mask된 항목은 미관측이라는 의미를 유지합니다.

`prepare_adamw`는 `pc_bootstrap`과 `pcv_preparation` recipe를 검사하고 parameter를
object identity로 한 번씩 분할합니다. P 또는 C 준비에는 공유 encoder·reader·후보
임베딩과 해당 private expert만 선택합니다. V 준비의 데이터 역할 이름 `verifier`는
모델의 `experts.validator`에 연결하며, **V private expert만** 선택하고 공유 세 영역과
P/C private expert를 모두 동결합니다. 동결은 선언 flag뿐 아니라 실제 `requires_grad`
상태로 확인합니다. 이 함수는 zero-step AdamW를 구성하며 update 명령은 제공하지 않습니다.

`ResumableSampler`, `capture_rng`/`restore_rng`, `save_preparation_checkpoint` 및
`load_preparation_checkpoint`는 role별 permutation·cursor, Python/NumPy/Torch CPU RNG,
명시한 장치의 RNG, recipe·dataset·split·구현 식별, 모델·zero-step AdamW 설정과 유한
usage를 보존하도록 준비되어 있습니다. checkpoint는 pending gradient가 없는 경계에서만
저장하며 `training_steps=0`과 optimizer state의 비어 있음을 확인합니다. immutable
checkpoint 안의 usage snapshot과 파일 저장·hash 완료 후의 **final usage receipt**를
구분하고, 재개에는 신뢰한 checkpoint SHA와 final receipt를 함께 요구합니다. 이 접점은
optimizer update를 실행한 학습 checkpoint의 인수를 대신하지 않습니다.

실제 수집 자료를 통한 유한 검사의 CLI는 `preparation_check.py`에 있습니다.
`COLLECTION_ROOT`와 아래 SHA는 원래 collection의 독립 등록 자료에서 가져와야 합니다.
다음은 실행 경로와 상한을 보여 주는 준비 예시이며 실제 dataset 검사 성공 기록은
총괄이 별도 확인한 뒤 연결합니다.

```bash
python -m rz_pals_model.preparation_check --collection "$COLLECTION_ROOT" --receipt-sha256 "$COLLECTION_RECEIPT_SHA256" --encoder-source-sha256 "$ENCODER_SOURCE_SHA256" --checkpoint "$OUTPUT_ROOT/initialization/untrained.pt" --output "$OUTPUT_ROOT/preparation/check.json" --max-records 64 --max-wall-time-ms 120000 --max-output-bytes 1048576 --max-input-bytes 134217728 --batch-size 1 --threads 1
```

이 검사는 P/C 기록을 한 번씩 소비하고 CPU FP32 forward·명시 mask·finite loss를 검사하며,
모든 parameter를 동결한 상태에서 전후 SHA가 같고 gradient가 없음을 확인합니다.
optimizer를 생성하지 않으며 backward·optimizer step·GPU 실행은 하지 않습니다.
입력·보고서 byte 및 준비·정리까지의 전체 벽시계 상한도 검사합니다. 값 target이 모두
미관측이어야 하는 collection에는 `--require-masked-value`를 명시해 그 조건을 확인할 수
있습니다. 현재 범위는 **단계 0의 준비이며 실제 학습은 미실행**입니다. 생성된 loss 숫자나
zero-step 재개 검사를 학습 진행·모델 개선·기력·성능의 증거로 해석하지 않습니다.

공개 K/V graph와 role graph를 나눈 것은 Rust 실행에서 공개 memory를 GPU에 유지할
접점입니다. Python reference는 CPU에서 검사하며 GPU resident KV나 물리 수명을
증명하지 않습니다. Rust runtime은 ORT I/O binding·worker lease·완료/격리 규칙을
별도로 연결해야 하며 node마다 Python callback을 호출하지 않습니다.

`rz_pals_model.verifier_producer`는 별도로 등록한 own collection의 공개 P/C 입력에서
학습 전용 V query를 만들고 **V forward와 CPU 실행 전에** 입력을 봉인합니다. frozen
미학습 V가 작업을 선택하면 자체 Rust `pals_cpu_task`에 최대 두 개의 유한 CPU 질문을
발주합니다. 제품의 V-free 경로에는 연결하지 않습니다. 원래 P/C 입력은 불변으로
보존하며 별도 private bank에 선택, 실제 CPU 조건·완료 깊이·중단 이유·비용과 후속
evidence를 기록하고 다시 읽어 검증합니다. 단일 작업 관측의 일곱 `preference_rank`는
모두 `None`이며 task target mask는 비어 있고 CE는 0입니다. 미실행 작업의 효용이나
WDL·전체 메이트 증명을 만들어 내지 않습니다. CPU threads는 2개, optimizer·backward는
없으며 parameter SHA가 전후 같아야 합니다. 문법·unit fixture와 실제 Rust dispatch
자료의 인수는 구분합니다.

중앙 `scripts/pals_model_validate.py`에 등록된 dataset과
`--verifier-cpu-binary`·`--verifier-cpu-binary-sha256`를 함께 주면 `resume_task` 하나에
대해 최대 8,192 CPU nodes·120초·4MiB·1 game·1 step의 실행과 private bank 재로딩을
검사합니다. 실제 자기 바이너리 SHA와 Rules 상태·이력·profile·검색 조건을 확인하며,
완료 iteration token이 없으면 실제 baseline과 unavailable 이유를 그대로 남깁니다.
이를 seven-way 효용 학습이나 기력 인수로 해석하지 않습니다.

`semantic_verifier.admit_semantic_input`은 기존 strict frozen current P/C parent와
별도 Rust `pals_cpu_task --prepare-semantic`의 원시 Rules 영수증을 연결하는 접점입니다.
결과 전 common query·등록 source/binary·before anchor·독립 launch 관측의 정확한
bytes와 pin을 함께 요구합니다. Rules가 준비한 board·이력 필드·합법 수·prefix·
restriction·claim의 의미만 private V 입력으로 쓰며 hash는 식별용입니다.
`FrozenSemanticVerifier`는 caller가 reload·동결한 기존 full V parameter를 보존하고,
CPU FP32로 seven-task logits를 반환합니다. 활성 autocast는 tensor 실행 전에 거절합니다.
11MiB의 추가 frontend 예약과 모델·checkpoint·전체 프로세스 메모리를 구분합니다.
이 접점은 새 private 의미 입력이며 기존 export 호환·학습된 V 품질을 주장하지 않습니다.

`verifier_utility.admit_fixed_obligation_utility`는 이 `CheckedSemanticInput` capability,
strict parent의 실제 prepared/tensor/lineage bytes, 사전 criterion·plan·declared prior
snapshot과 등록 checker의 실제 ResumeTask/Defer 원시 결과를 함께 소비합니다.
고정 H/N/wall/profile에서 동일 호출의 completed-iteration resume가 전체 H를 완료한
coverage와 정상 no-check Defer만 비교합니다. prior의 유효 completed iteration이
이미 H를 충족하면 새 양의 목표를 거절하고 partial·실패·취소·terminal은 mask합니다.
범위는 `declared_snapshot_relative_completed_iteration_coverage_only`이며 global
novelty·최적 선택·일반 seven-way rank나 기력 목표로 바꾸지 않습니다.
`frozen_verifier_utility_preparation`은 실제 semantic forward logits에서 별도 softplus
loss를 계산하고 parameter digest와 gradient 부재를 재검사합니다. optimizer·backward·
학습 update·제품 V 연결은 없습니다. 합성 fixture와 실제 child 실행 인수는
[PALS 구현·검증 기록](../../../docs/research/PALS-IMPLEMENTATION.md)에서 구분합니다.

`native_divergence`는 ordinary current dataset에 등록되지 않은 실제 C 이탈 입력의
별도 인수 접점입니다. `learning_input=false`인 producer journal, 원래 aux tensor·
lineage·source·physical event/raw output와 독립 export/launch pin을 검증하며 ordinary
parent는 같은 root의 Rules·이력·출처를 연결하는 anchor로만 사용합니다. parent의
public features나 current label을 aux에 복사하지 않습니다. raw Rules 의미 digest와
schema를 합성한 native encoding digest, export 당시 source와 실제 capture source는
각 provenance에 맞춰 따로 대조합니다. `shared_pc_if`의 scalar 역할 분기와 별도
P/C graph export를 명시적으로 지원하며 graph 이름만으로 역할을 추정하지 않습니다.

새 before-dispatch descriptor와, 과거 수집 입력에 새 Rust prefix Rules receipt를
연결한 `derived_from_legacy_prepared`는 다른 provenance입니다. 후자는 과거
descriptor 존재나 ordinary current label을 증명하지 않습니다. CPU collator는
batch 1~4·1MiB tensor 상한에서 captured divergence slot 순서를 유지하고 empty
candidates에는 한 개의 false-mask padding만 둡니다. 정책·WDL·task·divergence의
감독 mask는 모두 false입니다. 원래 native raw logits는 정답으로 사용하지 않습니다.
이 접점의 실제 frozen CPU C 대조도 ranking·수선 성공·학습·기력 인수와 구분합니다.

`repair_context`는 strict ordinary dataset의 현재 native Propose root와 최초
Repair decision을 연결합니다. 원래 true-learning prepared journal·tensor·lineage·
물리 완료·탐색 소비와 두 Rust Rules-only capability를 확인하고, counterexample의
첫 상대 응답이 proposal과 다르며 그 prefix가 실제 Repair 상태·이력·합법 수 순서에
도달하는지 검사합니다. `checked_existing_prepared_lineage`는 기존 기록을 현재
검사한 provenance이며 과거 before-dispatch descriptor로 표현하지 않습니다.

`bind_repair_candidate_pairs`는 같은 strict parent의 정확한 Repair current input에
대한 기존 proposer 후보 비교만 연결합니다. 부분 결과의 mask·관점·sign은 유지하고
all-masked 비교를 양의 준비 성공으로 바꾸지 않습니다. `frozen_repair_preparation`은
전후 CPU FP32·전체 module eval·gradient 부재·autocast 비활성·parameter digest를
검사하면서 기존 softplus ordinal loss를 소비합니다. 이번 목표는 다음 Repair 수의
조건부 ordinal 준비이며 전략적 수선 성공·counterexample 유효성·WDL·C divergence
ranking이나 학습 update를 인수하지 않습니다.

Public source JSONL은 다른 탐색에서 같은 observation을 다시 선택하면 동일한
원본 행을 반복 포함할 수 있습니다. D·Repair 소비자는 독립 observation SHA와
정확한 lexical bytes가 같은 행만 묶습니다. record ID·최신 revision으로 대체하지
않으며 다른 bytes는 원래 pin을 대신하지 못합니다. 전체 artifact pin과 요청·
journal·context·physical output의 유일성 검사는 유지합니다.

FLOPs 보고는 FMA=2 기준의 실제 선언 shape에 따른 matrix product 산술입니다.
encoder와 role 반복 비용을 구분하고 cached memory를 호출마다 중복 계산하지 않습니다.
softmax·RMSNorm·SwiGLU activation·lookup·이동·CPU 탐색과 backward는 제외 사실을
명시합니다. 이 수치를 전체 학습 FLOPs나 실행시간·VRAM 예상치로 대체하지 않습니다.
무제한 Colab 실행·Compute Unit 구매·실제 학습은 이 패키지의 실행 권한에 포함되지 않습니다.

참고: [공식 PyTorch 2.8 설치](https://pytorch.org/get-started/previous-versions/),
[ORT I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html).
