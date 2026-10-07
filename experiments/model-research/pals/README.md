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

공개 K/V graph와 role graph를 나눈 것은 Rust 실행에서 공개 memory를 GPU에 유지할
접점입니다. Python reference는 CPU에서 검사하며 GPU resident KV나 물리 수명을
증명하지 않습니다. Rust runtime은 ORT I/O binding·worker lease·완료/격리 규칙을
별도로 연결해야 하며 node마다 Python callback을 호출하지 않습니다.

FLOPs 보고는 FMA=2 기준의 실제 선언 shape에 따른 matrix product 산술입니다.
encoder와 role 반복 비용을 구분하고 cached memory를 호출마다 중복 계산하지 않습니다.
softmax·RMSNorm·SwiGLU activation·lookup·이동·CPU 탐색과 backward는 제외 사실을
명시합니다. 이 수치를 전체 학습 FLOPs나 실행시간·VRAM 예상치로 대체하지 않습니다.
무제한 Colab 실행·Compute Unit 구매·실제 학습은 이 패키지의 실행 권한에 포함되지 않습니다.

참고: [공식 PyTorch 2.8 설치](https://pytorch.org/get-started/previous-versions/),
[ORT I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html).
