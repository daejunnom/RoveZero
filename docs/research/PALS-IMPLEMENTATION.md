# PALS 구현 현황과 후속 인수 지시서

이 문서는 구현된 접점과 현재 실행 증거를 정리한다. PALS의 단계·V3 명세·비교 질문은
[기존 PALS 구현 계획](../PALS-IMPLEMENTATION.md)이 소유하며 여기서 새 계획으로 복제하지
않는다. 사용자가 제공한 `RoveZero_PALS_Internal_Algorithms_KO.md`와
`RoveZero_PALS_Architecture_Flows_KO.md`는 설계 자료다. 원문의 학습·실행 제안을 실제
승인으로 확대하지 않는다. 현재 목표는 **독자 Rust 엔진과 모델·학습 준비의 구현이며,
실제 optimizer 학습은 제외**한다. 생성 가중치는 미학습 초기화 자산이고 기력 인수는 없다.

Rust 제품의 실제 startup·역할 소비·취소·게임 수명·최종 증거 인수와 Python reference의
export·frozen forward·dataset preparation을 구분한다. Python 준비 성공을 Rust 제품의
실행 인수·학습·강도 결과로 승격하지 않는다.

## 처리 흐름과 소유 경계

제품 경로는 기존 UCI에서 탐색을 선택한 뒤 Rules의 정확한 상태·이력·합법 수를 사용한다.
`--search=puct`, `--search=cpu`, `--search=pals`는 별도 선택이며 PALS 작업을 기존 PUCT의
simulation·visit으로 해석하지 않는다. PALS는 P의 수순 제안 → 자체 CPU의 조건부 검사 →
C의 이탈 지점·반박 → P의 수선 → 관련 상황의 결론 갱신으로 진행한다. 마감 전 마지막
유효 착수를 선택하며, 모델의 추정은 Rules의 체크메이트·스테일메이트 판정을 대신하지 않는다.

| 소유자·소스 | 책임과 교체 경계 |
|---|---|
| Rules / [`rz-position`](../../crates/rz-position/src/lib.rs) | 합법 수, 특수 수, make/unmake, 완전 상태·이력·종료의 단일 소유자. |
| Own CPU / [`cpu.rs`](../../crates/rz-search/src/cpu.rs), [`cpu_value.rs`](../../crates/rz-search/src/cpu_value.rs) | CPU_T/CPU_R가 공유하는 유한 iterative deepening·PVS·aspiration·quiescence·TT와 평가 접점. profile·조건·depth·node·시간·완료 범위를 따로 식별한다. 초기 평가를 학습된 평가로 보고하지 않는다. |
| PALS Search / [`engine.rs`](../../crates/rz-search/src/pals/engine.rs) | 제안·반박·수선, 제한된 continuation 확장, CPU 작업 요청·소비와 루트 결정. `RoleModel`의 합법 후보 순서 logits·WDL 접점으로 모델을 시작 시 선택한다. |
| Canonical stores / [`store.rs`](../../crates/rz-search/src/pals/store.rs) | 상태·공유 line chunk·관측·상황·의존 관계·작업과 소비자별 완료를 보존한다. representation cache 회수와 검사 결과 삭제를 혼동하지 않는다. |
| PALS 계약 / [`pals.rs`](../../crates/rz-contracts/src/pals.rs) | `pals/0.1` 역할·authority·generation·situation handle·representation key·typed payload·CPU 조건. 총괄이 공통 계약을 소유한다. 기존 평가 계약 `0.1`의 가짜 `EvalOutput`으로 변환하지 않는다. |
| Runtime / [`rz-runtime/pals.rs`](../../crates/rz-runtime/src/pals.rs) | 큐·batch key·deadline·취소·물리 lease·typed 응답과 role-neutral/private memory namespace. 정상 물리 완료 전 입력·출력·session·workspace를 해제하거나 재사용하지 않는다. |
| 모델·ORT / [`pals_model.rs`](../../crates/rz-eval/src/pals_model.rs), [`pals_onnx.rs`](../../crates/rz-eval/src/pals_onnx.rs) | 입력·shape·dtype·유한값·예산·output 의미, 자산 pin, 공개 K/V 및 역할 graph, 하나의 물리 worker. cache hit·물리 NN 입력·탐색 소비를 별도로 기록한다. |
| 제품 조합 / [`pals_native.rs`](../../crates/rz-uci/src/pals_native.rs), [`main.rs`](../../crates/rz-uci/src/main.rs) | 정확한 Rules→모델 입력, 역할 요청→runtime, 모델 소비 ACK, UCI 시계·중단·새 게임·종료. 명시적 CPU/CUDA provider와 GPU startup 경계를 구현했다. CUDA-control 경로는 독립 pin의 typed inventory 및 실제 placement/profile witness를 요구하며 GPU 인수 상태는 아래 실행 기록과 구별한다. |
| 실행·데이터 / [`pals_manifest.rs`](../../crates/rz-experiments/src/pals_manifest.rs), [`pals_data.rs`](../../crates/rz-experiments/src/pals_data.rs), [`pals_collect.rs`](../../crates/rz-arena/src/pals_collect.rs) | V3 명세·lock·receipt, own-source 데이터·시점·입력 seal·분할·누출 검사, 수집 및 실패 보존. 외부 상대 엔진과 내부 모델 선택은 독립이다. |

CPU profile과 모델·입력의 의미가 달라지면 기존 작업 재개·cache 자격을 다시 검사한다.
CPU가 중단됐을 때 남은 completed depth와 frontier estimate를 구분하고, 미완료 결과를
요청 depth의 완료 bound로 저장하지 않는다. CPU와 WDL의 서로 다른 척도를 임의로 더하지
않는다. 외부 UCI 상대를 선택했다고 외부 엔진이 PALS 내부 CPU_R 구현으로 연결되는 것은
아니다. 실제 교체에는 capability·문제 조건·raw score·완료 범위의 별도 adapter가 필요하다.

## 원문 설계와 이번 구현의 범위

| 설계 항목 | 현재 구현·제한 |
|---|---|
| 학습 P/C/V+CPU_T, 실전 P/C+CPU_R | 모델의 P/C/V와 역할별 private parameter를 준비했다. 제품은 V-free P/C이며 실제 P/C/V 학습은 미실행이다. |
| 역할별 private latent와 공유 reader | 폭 384, Q/KV head 6/2, head dimension 64, private latent 16×384, reader 2 block×2회, SwiGLU 1024. board 64+metadata 2 토큰 encoder 2 block, record 내부 독립 encoder 1 block이다. |
| Canonical 사실과 GPU 표현 분리 | StateStore·LinePool·ObservationStore·SituationArena·TaskTable 및 의존 관계를 구현했다. GPU 표현을 회수해도 CPU 문제의 완료 기록은 보존한다. private latent를 공통 공개 기록으로 넣지 않는다. |
| 제한된 작업·저장소·중요 기록 | 모델 입력은 record 128·후보 256·이탈 지점 128 상한이며 required critical record 누락을 거부한다. 상황·line·관측·작업·큐·실행의 별도 유한 한도를 지킨다. |
| 실제 shared P/C reader 소유 | `public_memory`+`shared_pc_if` 두 session export와 native 소비 경로를 구현했다. 공유 reader·후보 임베딩 initializer는 outer scope 한 벌이며 private 초기 latent·4 FFN·head는 6개 ONNX `If`로 hard route한다. |
| 공개 K/V 재사용과 GPU 상주 | 역할 중립 key와 cache-on/off 수치 대조 접점, bounded host page bank 및 device K/V/I/O binding 경로를 준비했다. 현재 확인한 host page는 전체 입력 단위이며 record별 증분 인코딩이 아니다. 실제 device 상주·prepack 복제·VRAM 공유는 별도 인수 대상이다. |
| CPU/GPU 작업 겹치기와 private warm-start | 현재 native 역할 응답은 drain 후 반환하는 안전 경계다. CPU–GPU overlap 및 warm private latent의 의미·오차·효과는 미인수이며 현재 fresh 계산을 기준으로 남긴다. |
| 학습·resume·모델 교체 | own-source dataset·loss·recipe·zero-step AdamW·sampler/RNG/checkpoint·V-free export를 준비했다. 실제 CPU ORT P/C 수집과 별도 training-private V→CPU_T producer의 유한 실행을 확인했다. optimizer update 및 학습된 모델의 교체 일반화·강도는 미인수다. |

현재 가중치는 seed로 만든 **무작위 초기 파라미터**다. 실제 neural forward와 결정적
`legal-order-mock`은 별도 모델 종류로 식별하며 실패 시 서로 자동 대체하지 않는다.
신경망 실행이 된다는 사실만으로 유효한 체스 지식이나 개선된 착수 품질을 주장하지 않는다.

## 모델·입력·export 식별

구체 모델 소스와 명령은 [PALS 모델 패키지](../../experiments/model-research/pals/README.md)에
둔다. Torch 2.8.0·NumPy 2.2.6·ONNX 1.19.0·ORT 1.22.0·opset 17과 Rust
`ort 2.0.0-rc.10`의 실제 소비를 별도로 검사한다. Python은 offline 준비·export·독립 참조
용도이며 제품 탐색 노드마다 Python callback을 실행하지 않는다.

신형 artifact는 `rovezero.pals-model.v2`, `layout=shared_pc_if`, `layout_revision=1`이고
모델 의미는 v1이다. 기존 v1 separate P/C·P/C/V graph는 그대로 읽고 자동 변환하지 않는다.
제품용 `--rules-profile-json`은 실제 Rust encoder가 출력한 `rz-pals-rules-fields-v1`의
필드 순서·정규화·결측 의미를 pin한다. 58개 의미 문자열의 length-prefixed SHA,
source SHA와 선언 raw·canonical SHA를 기록한다. 의미 pin은 학습 입력 적합성의 증명이
아니므로 이를 별도 선언·데이터 검사 없이 학습 호환으로 승격하지 않는다.

P/C batch는 하나의 scalar 역할만 가지며 `If` condition은 CPU BOOL이다. 입력은 정확한
board·이력·record revision·모델 epoch·후보 순서·mask를 식별한다. 승격은 공통 Move16의
`0=none, 1=queen, 2=rook, 3=bishop, 4=knight`다. output은 요청 후보 순서와 차례 관점
WDL을 유지한다. P의 0 divergence padding은 의미 출력에서 제거한다. V 전용 head·latent가
제품 graph에 없음을 검사한다. raw logits·공개 K/V는 절대 `1e-4`+상대 `1e-3`, policy/WDL은
최대 절대 차이 `1e-4`로 독립 참조와 대조한다.

`reader_initializer_bank`의 byte·SHA와 branch-local copy 부재는 ONNX 직렬화 단일 소유의
증거다. native optimizer의 prepack·workspace 복제나 실제 VRAM 절약을 증명하지 않는다.
CPU `If` selected-only profile, 실제 native 수치·수명, GPU 실행과 메모리 관측을 구분한다.

## 자체 데이터와 zero-step 학습 준비

[`training.py`](../../experiments/model-research/pals/src/rz_pals_model/training.py)는
`rz-pals-data/2` 시점 고정 입력 seal과 native tensor sidecar의 byte SHA·encoder·history·
epoch·record 순서·critical·합법 후보를 대조한다. 독립 등록한 collection receipt와
source registry·split을 사용하며 target을 입력에 섞지 않는다. CPU raw 평가를 임의 WDL로
바꾸지 않고, policy·WDL·이탈·V 작업 각각의 명시 mask와 context를 요구한다.

PC bootstrap은 공유 encoder·reader·후보 임베딩과 선택한 P 또는 C private expert를
준비한다. V 단계는 V private expert만 선택하고 나머지 공유 및 P/C parameter를 실제
`requires_grad` 상태로 동결한다. zero-step AdamW parameter group·recipe·유한 비용,
role별 sampler permutation/cursor, Python·NumPy·Torch RNG, 모델과 optimizer 설정을
checkpoint에 보존한다. pending gradient가 없고 steps=0인 경계에서만 저장하며,
파일 저장·hash까지 닫은 final usage receipt를 재개 시 함께 요구한다.

[`preparation_check.py`](../../experiments/model-research/pals/src/rz_pals_model/preparation_check.py)는
등록된 실제 collection을 모두 한 번씩 소비하고 CPU frozen forward·masked loss·유한값·
parameter SHA 전후 일치를 확인한다. 이 CLI는 optimizer를 생성하지 않으며 backward·
optimizer step·GPU를 실행하지 않는다. 원시 collection·모델·receipt·보고서는 소스 밖
관리 루트에 두고, 공개 문서에서는 논리 경로와 식별만 사용한다.

[`pals_collect/native.rs`](../../crates/rz-arena/src/pals_collect/native.rs)는 실제 CPU ORT
P/C 호출의 준비 입력을 dispatch 전에 봉인하고, 물리 완료·raw 응답·논리 전달/거절·탐색
소비를 각각 남긴다. 독립 source registry의 binary·checkpoint·export·runtime·모델 구성·
encoding·epoch와 실제 자산을 대조한다. C 이탈 입력은 일반 합법 후보 head의 training row로
자동 변환하지 않고 별도 divergence sidecar와 계보로 보존한다. 가상 반박·수선에 실제
경기의 승패를 붙이지 않으며, backend가 raw 출력을 반환하기 전에 거부한 값은 오류 원인과
비관측 상태로 남긴다. raw가 관측됐다고 만들어 채우지 않는다.

[`verifier_producer.py`](../../experiments/model-research/pals/src/rz_pals_model/verifier_producer.py)는
등록된 공개 parent 입력을 재사용하되 별도 private query를 V forward와 CPU dispatch 전에
봉인한다. 자체 Rust CPU_T의 조건·완료 범위·비용과 후속 자료를 private bank에 보존한다.
제품 V를 활성화하거나 V private 상태를 P/C 입력에 넣지 않으며, 미실행 작업의 비교 순위·
정보 이득·WDL 목표를 만들지 않는다. 유한 producer 실행과 실제 V 학습은 별도 인수다.

## 현재 실행 증거

아래는 2026-10-07 문서 갱신 시점에 총괄이 확인한 실행 기록이다. 로컬 검사, 정확한 SHA의
CI 결과, 독립 등록 binary의 실제 실행과 dirty-source 모델 검사를 구분한다. reference 07의
기준은 `2e07757`+dirty source이며, reference 09는 `2d96a8a`+dirty source의 15개 파일 pin이다.
reference 09를 후속 최종 통합 SHA의 인수로 바꾸지 않는다. 후속 변경은 영향 검사를 다시 한다.
GPU 05~07의 등록 실행 기준은 `5bd8c59`, CPU paired 03의 등록 source는 `134bbe1`이다.
이전 전체 workspace 검사·후속 좁은 재검사·정확한 CI run과 등록 binary 실행을 구분한다.
`e604125`의 UCI pending-close 수정 직후 검사 전 기록은 보존하고, 후속 `134bbe1` 검사와
명시적 native loader 후보 `fc59b77`의 제한된 CPU 검사를 아래에서 별도로 기록한다.
후속 `bca916d`·`f78ecf0`의 독립 GPU 수치 검사와 `b952008`의 제품 startup 변경·CPU
검사·compile-only 등록도 별도 source·binary로 식별한다. 제품 재검사와 paired 대국의
남은 인수를 독립 checker의 정상 종료로 대신하지 않는다.

| 실제 자료 | 확인된 결과 | 해석의 한계 |
|---|---|---|
| 이전 Workspace CPU/mock 검사 | dirty-source에서 1,011개 통과, 16개 ignored. | 이전 실행으로 보존한다. ignored는 미실행이며 후속 변경의 검사 결과로 대체하지 않는다. |
| `975cce4` 로컬 CPU 검사 | workspace 1,061개 통과·16개 ignored, clippy 성공, default native CLI 4개 검사 통과. | 실제 CUDA·학습·대국 및 후속 SHA의 전체 CI 성공을 증명하지 않는다. |
| `975cce4` CI 및 `b04c886` 후속 수정 | 정확한 `975cce4`의 Linux·모델·CPU binding job 성공. Windows는 Unix 전용 fixture 2개 실패. `b04c886`에서 fixture를 수정했으며 해당 CI는 모델·binding 성공, Linux·Windows 실패다. | 해당 SHA의 실패 기록을 보존하며 `b04c886` CI 전체를 성공으로 표시하지 않는다. 후속 수정과 정확한 `5bd8c59`의 성공은 별도 행으로 기록한다. |
| `6979bdf` 전체 검사와 `5bd8c59` 후속 재검사 | `6979bdf` workspace 1,074개 통과·16개 ignored. 후속 `5bd8c59`의 optional CUDA launch payload boxing 2줄 수정에 대해 arena 69개 재검사와 전체 clippy 성공을 총괄이 확인했다. | 전체 workspace 실행 SHA와 좁은 재검사 SHA를 합치지 않는다. ignored는 미실행이며 GPU 종료·실제 학습·기력을 증명하지 않는다. |
| 정확한 `5bd8c59` CI | [CI run 37594871853](https://github.com/daejunnom/RoveZero/actions/runs/37594871853)의 Linux·Windows·모델·CPU bindings 4개 job 모두 성공을 총괄이 확인했다. | 해당 SHA의 CI 성공이다. 실제 CUDA 프로세스 종료, paired 대국과 source 밖 자료의 실행 인수를 대신하지 않는다. |
| `134bbe1` UCI 검사·CI | 후속 UCI all-feature 검사의 실행 결과 합계 249개 통과·workspace all-feature clippy 성공. [정확한 CI run 37597576640](https://github.com/daejunnom/RoveZero/actions/runs/37597576640)의 Linux·Windows·모델·CPU bindings 4개 job 모두 성공을 총괄이 확인했다. | `uci-final-08.log`와 `clippy-final-08.log`의 종료 0을 별도로 대조했다. GPU 05~07 종료 실패·paired 03 provider identity gate를 대신 인수하지 않는다. |
| `bca916d`·`f78ecf0` CI와 CPU 검사 | 총괄은 정확한 `bca916d` CI 4개 job 성공, `f78ecf0` all-feature 검사 1,092개 통과·16개 ignored·clippy/default UCI 검사 종료 0을 확인했다. [정확한 `f78ecf0` CI run 37604692688](https://github.com/daejunnom/RoveZero/actions/runs/37604692688)도 Linux·Windows·모델·CPU bindings 4개 job 모두 성공했다. | 각각의 source 검사다. GPU checker의 실행·제품 startup·paired 최종 저장은 별도 인수하며 ignored는 미실행으로 유지한다. |
| `binary-registration-06` | compiler-artifact JSON의 target·profile·features·binary SHA/크기와 `rz-uci-build-capability/1` metadata를 등록했다. source는 `5bd8c59`, UCI binary는 `7b283fea62af3518275b73bd53f6a15aadbb99d789f20f9ec04d4bcda7ab5a03`, CUDA numeric binary는 `a07495dec676c4fefd3d3bc6b367c746ee08395e6b46aced22181524968b4fd5`다. | compile capability만 확인한다. 영수증의 runtime/model loaded는 false이며, `--all-features` 명령 이력이나 source SHA만으로 다른 경로의 나중 바이너리를 같은 feature 빌드로 취급하지 않는다. |
| `model-reference-07` | wrapper 영수증 `success`, CLI exit 0. 모델 13개+학습 준비 20개, 총 33개 검사 통과. | 초기화 자산의 CPU 검사다. 실제 학습·GPU 결과가 아니다. |
| `actual-collection-01` frozen 검사 | 실제 9행 모두 한 번씩 소비. P 9/C 0. policy 4행 활성·5행 mask, WDL 9행 미관측 mask. parameter SHA 전후 동일, optimizer 미생성·backward 없음·steps 0. | 실제 C 조건부 collection, V context, 신경망 collector와 학습을 검증하지 않았다. C head 숫자 fixture와 실제 C 데이터는 별도다. |
| Native CPU PALS probe | 물리 완료 NN 입력 64개, 탐색 소비 32개를 구분한 유한 probe 통과. | 기력·처리량 향상이 아니다. 완료 입력을 모두 유효한 탐색 소비로 세지 않는다. |
| `cpu-numeric-20261007-975cce4-02` | 등록 source `975cce4`, exit 0·6개 참조 사례 통과. private repeat·cache/fresh 동등성·새 게임 cache/물리 ACK·shutdown 확인. 완료 NN 입력 20개 = public 6+role 14, 전체 3.331초, cgroup peak 284,164,096 bytes, OOM 0. 새 게임 뒤 host page reserved/pinned bytes·entries 모두 0. | CPU FP32 수치·수명 증거다. host page는 `whole_input`, allocator peak와 native resident parameter·prepack 공유는 unknown이다. GPU 성능·VRAM·강도 증거가 아니다. |
| `own-onnx-collection-01/own-onnx-01` | 독립 immutable collector binary `2444980f189b1b7d1d99ed9e71339492dded44bf268f1e7d6793df52d5756983`로 실제 CPU ORT NN 수집. 1게임·2 ply·6행, CPU 966 nodes/8 jobs, NN 완료 16개 = public 8+role 8, 역할 탐색 소비 8개. finish의 물리 shutdown·buffer 해제 확인, in-flight 0·quarantine 없음·학습 0. | 2 ply 제한의 결과는 unknown이고 value 6행 모두 mask다. C/Repair 및 divergence 계보를 실제 경기 승패나 전술 증명으로 채택하지 않는다. 독립 binary 실행을 최종 소스 SHA 검사로 합치지 않는다. |
| `model-reference-09` | 등록 collection 6행을 정확히 한 번씩 소비(P 4/C 2). policy 활성 0·WDL 활성 0·WDL mask 6, parameters 전후 동일·optimizer 미생성·backward 없음·steps 0. 별도 private V producer는 자체 CPU 신규 102 nodes, dispatch 1·실제 checks 2·후속 label 1; 부모 입력·weights 불변. | 현재 자료의 masked loss가 0이라는 결과는 학습 개선이 아니다. V의 비교 순위/task loss 목표는 0행이며 private-only, native 제품 V는 비활성이다. 기준 `2d96a8a`+dirty 15개 source pin과 최종 SHA를 구분한다. |
| GPU 시도 02 | 모델 실행 전 `native.missing_mapping`에서 실패. 실패 자료 보존. | PALS 신경망 CUDA 수치 실패로 해석하지 않는다. 입구 mapping 감사의 시점 결함은 후속 수정에서 분리했다. |
| Mapping 시점 수정 | dependency-only 입구 → 완료된 첫 native Run 이후 full audit. backend CPU seam 7개와 all-feature checker 빌드 통과. | 최종 audit 실패도 후속 실행을 막는다. CPU seam은 실제 CUDA 인수가 아니다. |
| GPU 시도 03 | 입구를 통과한 뒤 session 초기화에서 CPU EP 배정과 fallback 금지 충돌. 후속 네이티브 abort, exit `-6`. cgroup peak 3,305,578,496 bytes, OOM 0. | NN Run 이전 실패다. CPU 배정 노드와 종료 오류의 원인은 추가 확인 중이며 VRAM peak는 미관측이다. 기존 pin과 자원 한도를 조용히 바꾸지 않는다. |
| `gpu-numeric-04` | source `b04c886`, CUDA-control 실행 exit 1·77.625초, cgroup peak 3,397,308,416 bytes·OOM 0·cleanup 확인. 첫 NN Run 전 public 220노드(CPU 51/CUDA 169) 배치 gate 통과, shared P/C 453노드(CPU 61/CUDA 392) gate 거부. 등록 inventory와 CPU Gather 계열 4개 및 CUDA `MemcpyFromHost` 1개의 불일치를 보존했다. NN Run 0, 종료 후 GPU 사용 0으로 복귀를 총괄이 확인했다. | 해당 실행의 pre-Run 실패는 그대로 보존한다. 후속의 명시적 CUDA transfer 근거·등록과 05의 수치 보고서를 이 실패 기록에 소급하지 않는다. 임의 whitelist 확대·CPU NN fallback·inventory 재분류로 성공 처리하지 않는다. |
| `gpu-numeric-05` | 등록 source `5bd8c59`와 CUDA numeric binary로 실제 실행했다. `numeric.json`은 6개 CUDA FP32 사례·private repeat·cache/fresh·새 게임·물리 ACK·shutdown 통과를 기록했다. 완료 NN 입력 20개 = public 6+role 14. 그러나 보고서 저장 뒤 `malloc(): unsorted double linked list corrupted`로 프로세스 exit `-6`; execution은 failed, 전체 173.926초·cgroup peak 3,693,518,848 bytes·OOM 0·cleanup 확인·종료 뒤 GPU 사용 0을 총괄이 확인했다. | **GPU 실행 전체 인수는 실패다.** 수치 보고서의 완료/ACK와 프로세스의 정상 종료는 다른 증거다. VRAM peak·native allocator peak는 unknown이며 heap 오류 원인을 확정하지 않는다. CPU 자료나 후속 재실행으로 원래 실패를 덮지 않는다. |
| `gpu-teardown-06` GDB 진단 | 05와 같은 source·binary를 GDB의 기본 ASLR 비활성 조건에서 실행했다. stdout은 inferior의 정상 종료, stderr는 `No stack.`을 기록했다. execution의 GDB exit 0, 전체 141.735초·cgroup peak 3,840,204,800 bytes·OOM 0·cleanup 확인. | 상태는 `diagnostic_only_not_acceptance`다. 총괄은 진단 controller의 exit 1을 의도한 비인수 반환으로 확인했다. GDB·ASLR 조건이 달라 05의 수정이나 정상 실행 인수로 취급하지 않으며 heap 오류가 재현되지 않았다는 사실만 남긴다. 후속 07은 별도 행으로 기록한다. |
| `gpu-teardown-07-aslr` GDB 진단 | 05/06과 같은 `5bd8c59` checker binary·자산 pin에 ASLR을 켰다. 수치 보고서는 완료 NN 입력 20개 = public 6+role 14와 물리 ACK·shutdown을 기록했으나 inferior가 SIGABRT로 중단했다. stack에는 `libcudnn_engines_precompiled.so.9`와 `__run_exit_handlers`가 있고 stderr는 heap 손상을 기록했다. 전체 140.258초·cgroup peak 3,877,982,208 bytes·OOM 0·cleanup 확인. | GDB 자체 exit 0과 native inferior의 실패를 구분한다. 상태는 `diagnostic_only_not_acceptance`이며 **정상 GPU 종료 인수는 실패다.** 종료 중 cuDNN 호출 경로를 확인했지만 최초 메모리 손상 위치·책임자를 이 stack만으로 확정하지 않는다. |
| `fc59b77` 명시적 native loader 후보 | 총괄이 별도 명시 선택 후보를 커밋·push하고 all-target/all-feature CPU 검사 종료 0(eval unit 63개·loader 18개 및 해당 integration 검사), workspace all-feature clippy 종료 0을 확인했다. | 전체 test 수 합계는 주장하지 않는다. 이 CPU 검사 시점에는 새 후보의 GPU 실행이 인수 전이었다. 후속 08/09는 별도 행으로 연결하며 기본 선택을 유지한다. CPU 검사만으로 05/07 heap 오류의 수정·정상 native 종료를 주장하지 않는다. |
| `gpu-numeric-08-shim` 독립 checker | source `bca916d`·binary `1dee8ab061c60f78262460ccfbeb5ebb10f64fc244ad6b3d16e196898a873636`의 CUDA FP32 6사례·물리 NN 완료 20개(public 6+role 14)·runtime mapping ACK·물리 shutdown을 확인했다. execution passed·exit 0, 169.917초, cgroup peak 3,679,145,984 bytes·OOM 0·cleanup 확인. | 이 등록 조건의 독립 검사는 정상 종료했다. VRAM peak는 unknown이며 제품 UCI·arena GPU 실행 인수가 아니다. 이전 05/07 abort의 최초 원인을 확정하지 않는다. |
| `gpu-numeric-09-shim` 독립 checker | source `f78ecf0`·binary `f341e79407f2631d4b468b0cad0b208254abcc17ca5b5247dc34a8b04b76b28a`의 CUDA FP32 6사례·NN 완료 20개(public 6+role 14)·물리 ACK·shutdown 확인. execution passed·exit 0, 162.483초, cgroup peak 3,680,698,368 bytes·OOM 0·cleanup 확인. 총괄은 독립 메모리 검토 PASS도 확인했다. | 독립 GPU 수치·수명 검사의 PASS다. VRAM peak는 unknown이며 제품 startup·취소·새 게임·paired 대국을 대신 인수하지 않는다. 정상 실행을 과거 heap 손상의 최초 원인 제거 증명으로 해석하지 않는다. |
| `gpu-uci-smoke-09` 제품 startup | source `f78ecf0`·제품 binary `897c43cc17da8f79379098d623f8ac0c8139bd7687be8c35b1e7085a40a5434f`에서 `uciok`·ready 전에 stdout이 닫혔다. exit 2·97.414초, cgroup peak 3,600,363,520 bytes·OOM 0. stderr는 `PhysicalCompletionUnknown`, 물리 shutdown 실패와 startup mapping ACK `None`으로 인한 receipt publication 실패를 기록했다. | **제품 GPU startup 인수는 실패다.** transcript는 `uci` 송신 한 건이며 예정된 go 2회를 실제 실행으로 세지 않는다. 이 자료에서 정상 완료 NN 수·provider 영수증을 확정하지 않는다. 종료 뒤 GPU 0MiB 복귀는 총괄이 확인했으나 정상 NN 완료·startup과 다른 증거다. |
| 유한 runner 종료 검사 | clock+reap combined patch를 clean upstream에 적용한 별도 binary 빌드와 production-method syscall seam 14개 통과. | 실제 paired 대국은 별도 인수다. 원래 clock-only binary·patch·과거 자료는 보존한다. |
| `own-paired-cpu-01` 감사 정정 | 원래 입력·실행 자료를 수정하지 않은 append-only 정정으로, 복사한 UCI hash는 정확하지만 실제 binary에 `onnx-cpu` feature가 없었음을 확인했다. identification preflight에서 exit 1, provider·NN·readiness·게임 모두 시작 전이며 게임/NN 수는 0이다. | 대국·기력 실패로 집계하지 않는다. pair/process 영수증 생성 전 실패여서 실제 native cleanup 시간은 미관측이다. 결합 process/cleanup gate 문구만으로 물리 cleanup 실패라고 판정하지 않는다. |
| `own-paired-cpu-02` 준비 | `binary-registration-06` compiler proof와 compile-only capability를 입력 등록 전에 재확인했다. lock·opening·prepare-only 3개 명령 exit 0, private snapshot 9개를 준비했다. 준비 영수증의 engines started/NN ready는 false다. | 새 유한 CPU paired 기능 실행의 준비 자료다. 경기 실행·시계·PGN·정상 종료·강도 결과는 별도 실행 영수증이 도착할 때까지 미인수로 둔다. |
| `own-paired-cpu-02` 실제 preflight | 준비 이후 실제 readiness process는 exit 2·1.017초, `group_cleanup=gone`·errors 0으로 종료했다. stderr의 `UndeliveredDiagnostics`에는 PALS rounds·CPU nodes·소비 role 모두 0이다. supervisor의 service exit는 1, 회수 완료, 전체 16.343초 = service wait 10.808초+recovery 5.520초 등으로 기록했다. 경기 0·경기 PGN 없음. | readiness 실패이며 기력 대국 결과가 아니다. supervisor 시간은 실제 경기 시계·native cleanup 시간이 아니다. 회수된 process 영수증의 cleanup 276,164ns와 group 종료 증거를 별도로 보존한다. 준비된 opening PGN을 경기 PGN으로 세지 않는다. |
| `e604125` UCI 수정 | stop 이후 pending 착수·진단을 닫고 pipelined quit을 처리하는 후속 소스 수정을 총괄이 커밋·push했다. | 최초 기록 시점은 후속 root Cargo 검사 전이었다. 후속 `134bbe1` 검사·CI 성공은 위 별도 행으로 연결하며 이전 `5bd8c59`의 결과를 소급하지 않는다. 새 바이너리·readiness·paired 실행은 별도로 인수한다. |
| `own-paired-cpu-03` 실제 paired 실행 | source `134bbe1` 등록 binary로 120초+1초·흑백 교환 두 판을 실행했다. Rules PGN 감사에서 own CPU가 흑·백 모두 체크메이트로 승리했고 시계 감사 오류는 없다. Fastchess exit 0·group gone·cleanup 확인·process errors 0. supervisor service exit 1·회수 완료·전체 494.462초. | native provider 감사의 `native model/adapter/epoch identity differs` 때문에 integration gate가 거부됐다. 실행 영수증은 execution ready/strength eligible false·scored games 0이다. 두 판의 PGN·승패를 보존하되 정상 native integration 인수·Elo·모델 승격으로 사용하지 않는다. runner 종료만으로 NN provider identity·물리 종료를 대신 증명하지 않는다. |
| `own-paired-cpu-04` 실행과 append-only 감사 | source `bca916d`의 원래 production Debug에서 두 판의 Rules 체크메이트·시계·native provider gate 통과·scored 2를 관측했다. own CPU가 흑·백 모두 승리했다. 그러나 원래 operation은 `native evidence exceeds reserved bytes`로 최종 persistence exit 1이며 canonical pair receipt·Core가 없다. | 감사는 원본을 수정하거나 성공 영수증을 재생성해 저장하지 않았다. preflight부터 필수 최종 영수증까지의 전체 시간은 unknown이다. 64KiB 개별 pretty cap 대비 진단 재구성 67,622 bytes·초과 2,086 bytes는 Rust typed 재직렬화 미확인 값이며 전체 16MiB output cap 소진을 주장하지 않는다. 최종 저장·기력 인수는 실패 상태를 유지한다. |
| `own-paired-cpu-05` 실제 저장·Core 경계 | source `b952008`·등록 10의 CPU ORT 실행에서 흑백 두 판·Rules·시계·native integration gate를 통과했다. 첫 판은 PALS 백·own CPU 흑의 49 ply 3회 반복 무승부, 둘째 판은 own CPU 백의 37 ply 체크메이트 승리다. native receipt **68,102 bytes**가 2MiB 예약 안에서 실제 저장됐으며 PGN·Core 조립 진단도 보존했다. NN 완료 입력 12,266개·완료 role 6,133개·탐색 소비 role 6,123개, PALS session 2개의 물리 shutdown·buffer 해제·in-flight 0·quarantine 없음을 확인했다. | native 저장 수정의 실제 증거지만 **전체 실행은 exit 1**이다. Core 조립은 생성 PGN의 자유 서술 `source`가 공통 공개 HTTPS 계약과 충돌해 `core=null`·`artifact.source [InvalidSource]`로 거부됐다. 원래 실패·PGN·영수증을 변경하지 않는다. preflight부터 필수 native receipt까지 553,546ms, process cleanup 426,396ns, supervisor 전체 565.470초와 회수 3.810초는 서로 다른 관측 범위다. raw service peak 263,614,464 bytes는 Windows 전체 커밋·GPU·native allocator peak가 아니며 OOM 계수는 미관측이다. PALS failed search return 1과 취소·마감 진단도 원래 trace에 보존한다. |
| Core PGN producer 출처 연결 수정 | `project_pals_core_pgn`은 Core 참조의 `source`만 실제 locked Fastchess의 HTTPS 소스 URL로 투영한다. 원본 path·SHA·bytes·license·실행 설명과 producer 소스 commit·binary SHA를 `pgn_provenance`에 별도로 보존한다. 수정 파일 SHA는 `437e0a528bc72255e099ca1a4b9888f1b77aa0e9dca093a1a61ee3fbe2a479bd`다. 총괄은 arena all-target/all-feature CPU 검사 **221 pass·0 fail·14 ignored**, 기본 feature lib **55 pass·0 fail**, Clippy `-D warnings` exit 0과 독립 소스 검토를 확인했다. | URL은 이 로컬 PGN의 공개 다운로드 위치가 아니다. 공통 HTTPS 검증과 경기·시계·물리 수명·failure gate는 유지한다. 이 검사는 계약 fixture이며 CPU 05를 성공으로 다시 저장하지 않았다. 현재 production assembly는 live owner와 Serialize-only 증거를 요구하므로 typed 종단 replay 또는 수정된 새 paired 실행은 미실행·미인수다. 이를 JSON 재구성으로 대체하지 않는다. |
| GPU paired 06 준비의 후속 상태 | 기존 등록 10의 arena에는 위 Core 조립 결함이 있으므로 GPU 06 준비는 `superseded_prepared_not_run`·`execution_eligible=false`다. 원래 prepare/source proof는 불변으로 두고 별도 상태 파일에 새 arena 등록·새 실행 명세 필요를 기록했다. | GPU는 계속 `user-deferred`이며 새 GPU 실행은 없다. 오래된 준비 스크립트를 새 수정의 실제 실행 증거로 사용하지 않는다. |
| `f78ecf0` 증거 저장 예약과 `own-paired-gpu-05` 준비 | 고정 2MiB pair metadata 예약을 구현했다. GPU paired 05 준비 명령 2개 exit 0. 고유 input 3,143,597,770 bytes, input+output+runtime 필요량 3,227,483,850 bytes. artifact·failure recovery 각 4GiB, runtime 64MiB, output 16MiB 안에 pair metadata 2MiB를 예약했다. | execute command·engine·NN ready·GPU execution observed는 모두 false이며 GPU paired 경기는 **미실행**이다. cap·선언을 RAM/VRAM peak나 최종 저장 인수로 해석하지 않는다. CPU 04의 원래 실패를 소급해서 성공 처리하지 않는다. |
| `b952008` startup 변경·로컬 검사·CI | 실패 startup publication·진단·유한 단계 timing과 명시 CUDA startup probe 예산(1~180,000ms, 생략 기본 15,000ms)을 제품·manifest에 연결했다. 새 GPU 준비의 선택값은 120,000ms이며 실제 미실행이다. all-feature CPU 검사 1,103개 통과·16개 ignored, clippy와 default UCI·arena 검사 종료 0을 대조했다. CI의 중간 관측은 3개 성공·Windows 진행이었고, 총괄은 [정확한 CI run 37609418913](https://github.com/daejunnom/RoveZero/actions/runs/37609418913)의 최종 4개 job 성공을 확인했다(Windows 10:50:31 UTC 완료). | source 구현·CPU 검사·정확한 CI의 증거다. 새 제품 GPU startup·물리 수명·paired 저장 성공은 아직 인수 전이며 16개 ignored는 미실행이다. 새 예산 선언을 실제 실행시간이나 정상 완료로 바꾸지 않는다. |
| `binary-registration-10` | source `b952008`의 compiler-artifact 등록을 완료했다. `onnx_cpu`·`onnx_cuda`·experimental I/O binding compile feature와 CUDA path 컴파일을 기록했다. | compile-only다. runtime/model loaded와 GPU started는 false이며 모델 실행·provider ready·새 smoke의 성공 증거가 아니다. |
| `gpu-uci-smoke-10` 메모리 입구 | Windows `GetPerformanceInfo`의 커밋 여유 4,781,965,312 bytes가 등록된 최소 6GiB(6,442,450,944 bytes)보다 작아 admission이 거부됐다. systemd·GPU는 시작하지 않았다. 이후 사용자는 이번 세션 GPU 검증 보류를 선택했다. | 실제 CUDA 할당·startup 실패로 세지 않는다. 실행 중 Windows 커밋/VRAM peak는 미관측이다. 미실행·사용자 보류 상태를 성공·자동 재시도 근거로 바꾸지 않는다. |
| `gpu-uci-smoke-11` 실행 준비 | 총괄은 source `b952008`·등록 10·CAS를 고정한 바깥 launcher를 준비·freeze했다. probe 120,000ms·ready 180,000ms, 전체 240초 안에서 drain 최대 30초를 선언했다. | 준비 완료·실제 GPU 미실행·`user-deferred`다. 새 측정 조건·launcher 고정은 provider ready·NN 완료·정상 종료의 증거가 아니며 후속 사용자 재개와 메모리 입구 확인 뒤 실행 자료를 따로 인수한다. |
| 사용자 GPU 검증 보류 | 사용자가 이번 세션의 GPU 검증을 보류했다. 제품 smoke 11·최종 `b952008` source의 독립 numeric 10·새 GPU paired 06은 이번 세션에서 실제 실행하지 않는다. | 해당 항목은 `user-deferred` 후속 인수다. 준비·정적 검사와 과거 source의 08/09 PASS를 새 제품·최종 source 수치·paired 인수로 승격하지 않는다. CPU 검사·정확한 CI 4개 성공과 실제 학습 0은 유지한다. |

작은 실행·결과 JSON을 읽기 전용으로 대조한 자료의 논리 ID와 실제 파일 SHA-256은 다음과
같다. 원시 파일은 관리 루트의 `runs/pals/` 아래에 보존하며 개인 경로·command 원문·호스트
정보는 이 문서에 옮기지 않는다.

| 논리 ID (`runs/pals/` 기준) | SHA-256 |
|---|---|
| `cpu-numeric-20261007-975cce4-02/execution.json` | `561cdf38ad77ddd0236ed13cac4826002a2d0f322ba5d11035cb92da507abba4` |
| `cpu-numeric-20261007-975cce4-02/numeric.json` | `7ed25bd97c8db285684d3dbb6bbee0662f38a8e845f3cdde5c5d6fa4dfdef6dc` |
| `own-onnx-collection-01/own-onnx-01/receipt.json` | `a2a42c3786a7f9639e98c175b483c521af92fc35482e8464e6330c482b762e61` |
| `model-reference-09/validation.json` | `9257b60dd3c69a565be56473725e23bcafa5e9ff715dbfff8171d4ab28779eaf` |
| `model-reference-09/preparation-check.json` | `91984cd2d841ac6433ec93526029d0483cb0ac3c792469ab8014673fc24d4394` |
| `model-reference-09/private-verifier/receipt.json` | `8f9227c5c34b75fa6e915c99c63cce7513c483f195bcfa13cc87d445bbe7ab4f` |
| `gpu-numeric-04/execution.json` | `1453a0ca1acd9a0196f0f79d1fd1e4b6fab2f4d6bd149b5e94da27eb22f14859` |
| `gpu-numeric-04/numeric.cuda-control-primary/public-initial-placement.json` | `0d60ee96e4aa2bfd78eb3c5395e9005bde7820874c316352e79300e4a12232a7` |
| `gpu-numeric-04/numeric.cuda-control-primary/shared_pc-initial-placement.json` | `7192905c844af038dbb5cc80ac1b8f6fc1fe342b98b3986158542c69c4db9fdd` |
| `gpu-numeric-05/execution.json` | `7e2a766401fb4adb917955c0284dd1bbeeae78d9b125f53d9368c5fcbf4ffcd1` |
| `gpu-numeric-05/numeric.json` | `5921a80c2f5982cd8a90a2a8797b90c292c28f3c555110108c00ede884e3b3c2` |
| `gpu-numeric-05/stderr.log` | `2fec9196496bf208222bf35ce8032d8968a407dd34dcce54135658bc493b7463` |
| `gpu-teardown-06/execution.json` | `a5e7d7ea94739902128f49dc9e576c0e129404c91540fe27b8f67ac89e915d2b` |
| `gpu-teardown-06/stdout.log` | `b4d43b3c5775f47920f8edc1091c39d554f884a281eb959c730eee3e3668d140` |
| `gpu-teardown-06/stderr.log` | `49e1116d47bda20a27ba350b585760440aaf899c435196f894edfc207c294ea0` |
| `gpu-teardown-07-aslr/execution.json` | `6de9596b97fc4c61949a958a92cfb6fffb0994a9f5c50afbcbc2362ee83cee5c` |
| `gpu-teardown-07-aslr/numeric.json` | `25bdb7502e87dd41f3a706eb2302ef92f2885f409faa90f23452ee2fb83bceb9` |
| `gpu-teardown-07-aslr/stdout.log` | `4e461f99f68c4833b6913e6cf34e4051ed7c52a855282f6e90a7d1e546169ad0` |
| `gpu-teardown-07-aslr/stderr.log` | `8c621c067a186e39cf0ccbdac91566da29eb0e72149c1f398801907f9cbaeb76` |
| `gpu-numeric-08-shim/execution.json` | `78a06c96c02aff7df136dda0ade0f71123dca1d1c0c637302a47e3d5251189ac` |
| `gpu-numeric-08-shim/numeric.json` | `53fa4c164d9521f0c82c5290c3aceb53818ac227852e85179c466bc3c90fe071` |
| `gpu-numeric-09-shim/execution.json` | `ab03cf96f2ce3fadeb5e8729adf11c4e9a937af0e82361bca058c0bed75d6704` |
| `gpu-numeric-09-shim/numeric.json` | `48bf0e8dbb68c89d090de8b287eadbcf4aeb0abc7a79407a228365e4208c5fa7` |
| `gpu-uci-smoke-09/execution.json` | `51eebc8551946eea3ea3bc45dec5a1c452bb177b07ab04a8fa1cf623d5c60fc8` |
| `gpu-uci-smoke-09/stderr.log` | `ed9336fa078865ac1beee592e9bd9c05a655b76217331d0ee5cbff975ac7ff97` |
| `gpu-uci-smoke-09/transcript.json` | `cb0355b3571d41feb7f394c69b7a01ce389f29037660fb17e99723fc3e585639` |
| `binary-registration-06/registration.json` | `fad9dc73fddb2ea089d463a612869a4e6ca73d43eac9bf71502fa38335e594c2` |
| `binary-registration-06/build-capabilities.json` | `4cef04dfa2e6560226c2453fc0cdfb69afa49661608f0cb3bbb5a43c571e96ad` |
| `binary-registration-10/registration.json` | `b17293546dd33c419d5ccdd4861e7fec6e3a9e2d94bcef5a8c1f6772dd11012d` |
| `binary-registration-10/build-capabilities.json` | `4cef04dfa2e6560226c2453fc0cdfb69afa49661608f0cb3bbb5a43c571e96ad` |
| `own-paired-cpu-01/audit-correction-01.json` | `99cc4d96bd73ecd154228526c7035843294d51101a1c098b0265ba81c50a19c0` |
| `own-paired-cpu-02/preparation-receipt.json` | `19e331e67bb6cc4b79d815c65c9fc5ffac46f64095c0225ae60a8b262696c7fc` |
| `own-paired-cpu-02/execution-result.json` | `4cd480a94e6300a6bf139f4c1b959318075ddf8f8bf02c1c606d2790a2932d94` |
| `own-paired-cpu-02/recovered/attempt-01/external-baseline-readiness-process.v2.json` | `ecacdae96c69c2f96c12fe564830bb035971bf01e3c8822f7772065ded8d55f4` |
| `own-paired-cpu-02/recovered/attempt-01/external-baseline-readiness.stderr.log` | `23d48e310be136adae0fc8d067c6a2b8b351659c32317e931d158ac97cb9b9ce` |
| `own-paired-cpu-03/execution-result.json` | `ffa9dafedc3641f73d94b7b861776433cf7d50d7894e2d420d2118f015d3f363` |
| `own-paired-cpu-03/recovered/attempt-01/pals-arena-pair-receipt.v3.json` | `8f7c8002cd1e5c5aff8d5d84f97f9cb4ab254c663e5f7ef4e29997334955a2da` |
| `own-paired-cpu-03/recovered/attempt-01/match.pgn` | `96cabf2d1642d7e2be325b7816f923a41ecbc9b673b296ae0793f1d9ab65e0d0` |
| `own-paired-cpu-04/audit-persistence-failure-01.json` | `8251d84cc6c54e6733309a8cd5ba3ff9efccfb438093e666820c1caf96baa5e6` |
| `own-paired-cpu-05/execution-result.json` | `5a882e43b5db9fa382e5c18333ba1448f294fdfc3329822c35a86d1f55079aa6` |
| `own-paired-cpu-05/recovered/attempt-01/pals-arena-pair-receipt.v3.json` | `2cf19f0bf36f67e41e87715dcc6ee4a8c7459a7a91ca6e75ff5248749a58558e` |
| `own-paired-cpu-05/recovered/attempt-01/pals-core-assembly.v3.json` | `e753eb40400485f429f2cf3da9546ba11a70895e4e1eca85f1c3bfd6bc7fe226` |
| `own-paired-cpu-05/recovered/attempt-01/match.pgn` | `0afaad928f9b7f81bbe209cd5f827a25f0c722c2a5c20468f768730b80888f2f` |
| `own-paired-cpu-05/observed-audit-01.json` | `83f647de93e604670c1dabfc3608602c21628d610433e7f4fb15d8b5d08b032f` |
| `core-provenance-check-01/checks.json` | `2e150c34c862e559b868d58e55d27e6a82ef4427bc52d073d88d522031918531` |
| `own-paired-gpu-06/superseded-01.json` | `6fc33f3da9acc77221ed6e167439d0d32f02cc9a900772f1f89ef34ea2a248b7` |
| `own-paired-gpu-05/preparation-receipt.json` | `91528fd5b8be3c6bb1a693c34894ba2cf82c9ca46400cdd58270da2c31b9dced` |
| `gpu-uci-smoke-10/windows-commit-preflight.json` | `ba2ae8c58df480787aa261f09dd01dd5c99b7967053df83ae5389362776ebe35` |
| `workspace-check-03/uci-final-08.log` | `80ffbd12a9a30fb539719aafe9432766e0ae7a851f9398c5702e090b25c781a4` |
| `workspace-check-03/clippy-final-08.log` | `da3ea705553fbf108819b87159fbbebcbfaa02c2a277f7759fe59d823761905b` |
| `workspace-check-03/tests-startup-11.log` | `f18f4c1c3124984b92cf945baea7ef74a924a2f636b0bd084e426d55f32e390d` |
| `workspace-check-03/clippy-startup-11.log` | `59de3c77f10ebcd6e216f7137279adcd0faf20406f27967dd45bb2019176a360` |
| `workspace-check-03/default-startup-11.log` | `30c85c88ef3cb6a15ba22311e2050f490f6ab85af1bed63e610fdaaa50aa6248` |

정확한 `b952008`의 최종 CI 조회는 관리 루트의 `reports/pals/ci-b952008-final.json`에
보존했다. 직접 대조한 head SHA는 `b95200864825b2e88ca46ef86dd23ba6ff7a20a5`, 상태는
completed/success이며 4개 job이 모두 성공했다. 이 작은 조회 영수증의 SHA-256은
`97e11a1ca99e259e31229bd86181078a8b064e8868d8037e3614b15c12b17599`다.

문서만 갱신한 `0b47000d2192a7862f6274e3f0c834fd1acec728`의
[CI run 37613007601](https://github.com/daejunnom/RoveZero/actions/runs/37613007601)도
4개 job 모두 completed/success를 확인했다. 조회 영수증은
`reports/pals/ci-0b47000-final.json`에 보존하며 SHA-256은
`dc236278403e7001399e987c6bc30d3500936b57d76e3e3899533ced4b4138c9`다.
이 CI와 CPU 05의 실제 실행 source `b952008`·등록 10은 구분한다.

## 남은 인수와 진행 순서

| 순서 | 필요한 확인 | 현재 상태·책임 |
|---|---|---|
| 1 | 독립 checker의 GPU 수치·mapping·물리 shutdown과 과거 heap 실패 경계 | 08/09 독립 checker의 정상 exit 0·수치/물리 ACK PASS를 확인했다. 이전 02~07 실패·GDB 조건은 보존하며 05/07의 최초 손상 위치는 확정하지 않는다. 현재 제품 기본 선택과 명시 profile을 구분하고 단독 결과를 제품 startup 인수로 확대하지 않는다. |
| 2 | 제품 UCI의 startup·GPU 수치·취소·늦은 완료·drain·buffer 수명 | 제품 smoke 09는 `PhysicalCompletionUnknown`과 실패 영수증 publication에서 실패했다. `b952008`에서 진단·유한 timing·명시 startup 예산 지원을 구현하고 검사했다. 현재 GPU 준비값은 120초이며 생략 기본값은 15초다. smoke 10은 Windows 커밋 입구에서 GPU 시작 전 거부됐다. 새 smoke 11은 준비 완료·미실행·`user-deferred`이며 후속 실제 startup·물리 수명 자료가 필요하다. |
| 3 | 최종 source 독립 GPU 수치·제품 자원·실패 증거의 저장 | 등록 10은 compile-only다. 이전 08/09의 cgroup peak·OOM 0을 기록했으나 VRAM peak는 unknown이다. 최종 `b952008`의 numeric 10과 새 제품 자원·startup/ready/종료·실패 영수증·저장은 `user-deferred`다. 이번 세션에는 새 GPU 실행을 하지 않고 후속 인수에서 등록 조건·메모리 입구를 다시 확인한다. |
| 4 | 실제 NN 기반 own collection 및 C 이탈·V task context, frozen epoch·mask·split·누출 | CPU ORT collector 6행과 frozen preparation 전체 소비, 별도 private V→CPU_T 유한 producer 실행을 확인했다. GPU collector, C divergence head의 명시 학습 context, 유효한 정책·결과·작업 효용 목표와 더 넓은 split/holdout 자료는 별도 인수다. |
| 5 | 후속 record별 증분 인코딩·device warm-start·CPU/GPU overlap과 실제 공유 메모리 실험 | 확인한 host bank는 whole-input cache이며 현재 fresh 역할 계산을 기준으로 한다. record별 증분 인코딩·warm latent·overlap과 native prepack/VRAM 효과는 후속 별도 실험·미인수다. 의미 보존 E와 근사 A·스케줄 S를 구분하고 새 고정 작업량·비교 질문·자원·peak 관측을 등록한다. |
| 6 | V-free PALS+Own CPU_R의 유한 paired 실행·시계·PGN·native 최종 receipt/Core | CPU 01/02 입구 실패와 03 identity gate 실패를 보존했다. CPU 04는 경기·native gate를 통과했지만 최종 persistence 실패·receipt/Core 부재다. CPU 05는 2MiB 예약에서 native receipt 68,102 bytes 저장을 실제 확인했으나 Core PGN 출처 계약 오류로 전체 exit 1이다. 원래 자료를 보존하며 생성 증거와 공개 producer 출처의 접점을 따로 검증한다. GPU paired 05 metadata 준비 뒤 경기는 미실행이고 새 GPU paired 06은 `user-deferred`다. 원래 실패·unknown 시간을 덮지 않고 두 판으로 Elo를 확정하지 않는다. |
| 7 | 정확한 통합 SHA·CPU CI·독립 GPU·제품·paired 인수 연결 | 이전 source 검사와 `f78ecf0`/`b952008` 정확한 CI 4개 성공, `b952008` CPU 1,103개·clippy/default 종료 0을 연결했다. 08/09 독립 GPU PASS와 제품 09 실패·10 GPU 미시작, CPU 04 저장 실패를 보존한다. 새 제품 11·최종 source numeric 10·GPU paired 06은 `user-deferred`이며 정적 준비를 실제 인수로 승격하지 않는다. |

능동 CPU 대체 응수의 후보·반박·수선 연결, evaluator identity를 포함한 근거 namespace,
실제 착수 뒤의 완료 근거·paused 작업 재개, CPU adapter 교체 경계도 별도로 인수한다.
노드가 남아 있다는 사실을 새 root에서 작업을 유효하게 재개했다는 증거로 바꾸지 않는다.
실제 NN collector와 학습 전용 V→CPU_T 유한 dispatch producer는 구현했고 위 제한된 CPU
실행 자료를 확보했다. 기존 CPU/mock collection이나 loss fixture를 그 실행 자료로 바꾸지
않는다. NN 물리 완료·role 소비·CPU 신규 작업·실전 경기 결과·학습 목표의 관측 범위가 다르며,
모든 목표가 mask인 frozen forward 성공을 학습 준비 데이터의 유용성이나 기력으로 표시하지 않는다.

GPU 05의 heap 오류는 보고서의 수치·물리 ACK만으로 원인을 확정할 수 없다. 등록된 source·
binary·자산·runtime·placement 조건과 실패 자료를 보존하고, teardown의 실제 경계와 오류
발생 시점을 좁혀 확인한다. GDB·ASLR 변경 후의 미재현도 원인 제거를 증명하지 않으며,
07의 cuDNN exit-handler stack도 최초 손상 위치를 확정하지 않는다. 명시적 loader 조건의
독립 checker 08/09는 정상 종료했지만 그 사실만으로 원래 heap 손상 원인 제거·제품 startup·
모든 실행 경로의 정상 수명을 확정하지 않는다.
조건 변경·CPU fallback·무한 재시도로 원래 실행을 성공 처리하지 않는다.

실제 optimizer 학습은 이 목표의 남은 필수 실행에 포함하지 않는다. 이후 학습을 진행할
경우 own-source 데이터·권리·예산·role 순서·frozen epoch·holdout과 checkpoint 조건을
다시 확정한다. 공개 모델 자산이 아직 미학습이라는 사실은 문서와 실행 영수증에 유지한다.

PALS는 기존 PUCT·LC0와 모델·탐색을 바꾸는 구현이다. **자료형 변경 등 의미 보존 변경에
사용한 시간·메모리 5% 문턱은 이번 PALS 도입의 진행·통합 조건이 아니다.** 후속 의미 보존
E 최적화가 필요하면 해당 변경의 baseline·peak·작업량·채택 기준을 별도로 등록한다.
현재 비교는 선언한 모델·탐색·runtime 전체 또는 통제한 단일 축의 질문에 맞춰 해석한다.

## 교체와 복구 지시

모델·Own CPU profile·외부 상대를 각각 독립적으로 선택한다. 새 모델은 typed 역할 입력·
후보 순서·WDL·head·cache 의미와 실제 자산 pin을 먼저 검사한다. 새 CPU는 지원 질문·
조건·완료 범위·score 관점과 재개 capability를 맞춘다. 외부 상대의 UCI·옵션·시계·process
종료는 arena가 소유하고 PALS 내부 입력이나 역할 head를 요구하지 않는다.

미지원 형식·계약·provider·pin 불일치, 할당·물리 완료 실패가 발생하면 명확한 오류로
중단하고 실패 자료를 남긴다. session·buffer 완료가 불명확하면 격리하고 새 요청에
재사용하지 않는다. cache 재계산은 canonical 기록과 동일 입력에서 수행하며, cache
회수가 CPU 검사 재실행이나 완료된 사실 손실을 요구하지 않도록 검증한다.

복구는 등록된 이전 바이너리·자산·profile 및 기존 V1/V2·PUCT 또는 Own CPU의 명시적
실행 선택으로 한다. PALS 오류 뒤 숨은 mock·LC0·Stockfish fallback으로 정상 성공을
만들지 않는다. 계약·정확성·수명·대국 증거와 다른 작업의 WIP를 보존하며 총괄이 인수한다.

## 2026-10-07 후속 진행 — CPU06 실패와 CPU 준비 수정

이 절은 위 기록을 보존한 후속 상태다. `30f2a0d`의 완료 감사와 등록 11의 실제 CPU06,
후속 개별 수정·CPU 모델 준비 검사를 서로 구분한다. **전체 목표는 아직 미완료이며,
CPU06은 integration 실패, GPU 검증은 사용자 보류 상태다.** 아래 좁은 검사나 원시
프로세스 종료 성공을 최종 Core·대국·GPU·학습 인수로 바꾸지 않는다.

### CPU06의 실제 결과와 두 실패 경계

`own-paired-cpu-06`은 등록 11의 source
`30f2a0dc6d49a79fd73fb27adc84209ea0b60a53`로 실행했다. Ponder off, 엔진별 선언
CPU 2·동일 affinity `0,2`, GPU 없음, 120초+1초 피셔, 흑백 교환 두 판의 기존 조건을
유지했다. `failure-review-02.json`과 supervisor 결과를 읽기 전용으로 대조했다.

| 관측 | 실제 결과 | 인수 범위 |
|---|---|---|
| runner 프로세스·정리 | native process exit 0, process group `gone`, `cleanup_verified=true`, unresolved owner 없음. | runner 종료·정리 증거다. 서비스 전체·NN·시계·Core 성공과 동일하지 않다. |
| 서비스·자료 보존 | service exit 1, `recovery_complete=true`, `integration_checks_passed=false`. native pair 영수증과 원래 PGN·로그를 보존했다. | 실패 실행과 회수 성공을 함께 기록한다. 원본 실패를 소급 성공으로 바꾸지 않는다. |
| 첫 경기 | 백 `pals-onnx-cpu`, 흑 `own-cpu`, PGN 결과 `0-1`, PALS 백의 시간패 초과 5ms. | 실제 패배 기록이다. 두 번째 판과 함께 보존하며 표본 부족이나 감사 오류를 이유로 유리하게 누락하지 않는다. |
| 두 번째 경기 | 백 `own-cpu`, 흑 `pals-onnx-cpu`, PGN 결과 `1-0`, PALS 흑의 시간패 초과 7ms. | 실제 패배 기록이다. pilot 두 판으로 Elo·모델·탐색 승격을 확정하지 않는다. |
| PGN 감사·Core | `PGN engine loss winner disagrees with A side to move`, `scored_games=0`, Core collection `not_reached`. | 정식 집계·Core 인수가 이 실행에서 이뤄지지 않았다는 뜻이다. 두 시간패가 없었다거나 무승부였다는 뜻이 아니다. |

고정 Fastchess는 반환된 후보를 PGN에 먼저 기록한 뒤 시간패를 검사하며, 시간패 후보는
실제 board에 적용하지 않는다. 현재 arena의 재생은 마지막 늦은 후보를 실제 적용수로
읽어 실패 색과 차례를 뒤집었다. 이는 PGN 감사의 별도 정확성 문제다. **native full-clock
감사의 `engine_loss` 거부는 실제 시간패에 대한 유효한 별도 판정**으로 보존한다. PGN
감사를 수정해도 5ms·7ms 시간패가 사라지거나 full-clock·Core가 통과한 것으로 표시하지
않는다. 원시 PGN을 덮어쓰지 않고 실제 적용수와 기록된 늦은 후보의 경계를 감사한다.

### 후속 수정과 CPU 모델 준비의 확인 범위

아래 검사 수는 총괄이 확인한 개별 실행 범위다. `model-reference-10/validation.json`은
CPU provider·상태·source pin을 직접 대조했으나 개별 unittest 개수는 담지 않는다.
clean `30f2a0d` CI와 dirty-source 모델 검사, 후속 커밋의 좁은 검사를 합쳐 같은 통합 SHA의
전체 성공으로 표시하지 않는다. 이번 절은 후속 HEAD의 새 CI 성공을 선언하지 않는다.

| 변경·자료 | 확인된 구현·검사 | 남은 경계 |
|---|---|---|
| `dcb312c` | Rules의 정확한 상태·이력을 사용하는 유한 UCI 수순 재생 접점을 추가했다. 총괄이 CPU 검사 52개 통과·2개 ignored를 확인했다. | ignored는 미실행이다. 외부 checker의 실제 소비·실행 인수를 대신하지 않는다. |
| `02b01a8` | RulesTerminal 체크메이트의 승자와 차례 일관성(MF-01), 잔여 예산에 따른 collection 실제 읽기 제한(MF-02)을 수정했다. 총괄이 Rust data 검사 21개 통과를 확인했다. | 정상 ActualGame 후속 결과와 RulesTerminal을 구분한다. 새 모델·전체 학습 준비·paired 인수로 확대하지 않는다. |
| `361c782` | 성공한 CPU_T capture 뒤 quota·저장 실패에도 조건·실제 반환·원래 오류를 회수하도록 MF-03 보존 경계를 수정했다. | source 수정과 해당 실패 보존 검사를 실제 학습·GPU 또는 모든 producer 경로의 종단 성공으로 바꾸지 않는다. |
| `model-reference-10` | source `30f2a0d`+dirty의 개별 source 파일 pin으로 CPU unittest 58개, 미학습 `shared_pc_if` export와 독립 CPU 수치 검사를 총괄이 확인했다. 소형 영수증은 `status=success`, provider CPU, 학습 0 step, GPU `not_run`이다. | 초기 파라미터의 CPU 준비 증거다. source pin 없이 clean `30f2a0d` CI나 후속 커밋 전체에 재사용하지 않으며 학습·기력·제품 GPU 성공으로 승격하지 않는다. |
| `4a8a446` checker 교체 경계 | 자체·외부 report를 분리한 인터페이스와 유한 Linux UCI owner를 추가했다. 첫 search 전체 171개와 후속 checker 집중 16개가 통과했다. 후속 정적 검사의 Copy 정리도 통과했다. | 집중 검사는 이전 전체 검사의 일부와 겹치므로 개수를 더해 고유 검사 수로 표시하지 않는다. 외부 CPU_R의 PALS 탐색·제품 CLI·receipt 소비는 아직 미완료다. |
| `e330c7c` PALS 작업 마감 | 기존 soft/admission 중 이른 시각을 작업 deadline으로 전달했다. 해당 소비자 회귀를 포함한 UCI lib 133개가 통과했다. 전역 시계·물리 완료 fence는 유지했다. | 실제 두 판에서 시간패가 해소됐는지는 새 등록·실행 전까지 미검증이다. |
| `5f9f037` timeout PGN 감사 | 실제 적용수와 기록된 늦은 후보를 구분했다. PGN 29개와 native pilot 5개가 통과했다. 영향 패키지의 Clippy `-D warnings`와 workspace fmt 검사도 통과했다. | 기존 CPU06 실패를 성공·Core로 바꾸지 않는다. 새 필드는 failure-only이며 원시 PGN은 보존한다. |

### 전체 목표에 남은 구현과 별도 실험

`completion-audit-30f2a0d-02.md`의 감사 시점은 `30f2a0d`다. 감사 뒤 수정과 실행은
위처럼 추가 기록하며, 감사의 구현 공백을 과거 검사·모델 fixture로 소급 완료 처리하지
않는다. 실제 optimizer 학습 제외는 데이터·계약·학습 준비 API의 미구현을 제외한다는
뜻이 아니다.

| 항목 | 남은 구현·인수 |
|---|---|
| 외부 CPU_R checker 소비 | 대국 상대 UCI 연결·`CpuSearcher` trait·mock을 넘어, 실제 문제 조건·capability·authority·raw CP/WDL/mate/bound·deadline·generation·`stop`/`bestmove`·프로세스 종료와 실패를 내부 checker가 소비하는 경로를 인수한다. 현재 WIP를 완료로 표시하지 않는다. |
| record별 공개 K/V | 현재 whole-input cache와 독립 record encoder를 구분한다. record/chunk delta→새 page→기존 page 재사용→native join·pin·evict의 실제 소비 경로와, 표현 회수 후 완료 CPU 근거 보존을 구현·검사한다. page key 타입만으로 구현 완료를 주장하지 않는다. |
| 역할별 private warm-start | 현재 fresh 역할 latent를 기준으로 남긴다. 다음 query의 역할 상태 재사용 경로와 fresh/warm 의미·오차·실제 비용 대조는 미인수다. warm-start·device 상주·압축·overlap의 효과는 후속 A/S/E 비교이며 구조적 준비와 연구 성과를 구분한다. |
| P 동일 시작 상태 비교(DG-01) | 같은 시작 상태의 후보를 묶는 pair/context, 별도 comparison target·collation·loss API가 남는다. 일반 policy loss를 이 비교의 대체 구현으로 표시하지 않는다. |
| C native divergence(DG-02) | `pals_collect/native.rs`의 `NativePreparedContext::Divergence`가 실제 divergence 입력과 sidecar에 capture하는 경로는 존재한다. 현재 `training_admission=deferred_divergence_head`, `counterfactual_wdl=masked`이며, 남은 것은 전용 감독 context/envelope·loader 입장·target/ranking 연결이다. 별도 capture와 조건부 BCE를 후보 ranking 완료로 해석하지 않는다. |
| V 비교 감독·문제 입력(DG-03/04) | 실제 조건부 관측의 comparative utility target producer·유효한 owned ranking 자료, 같은 parent·예산에서도 다른 branch/question을 구별하는 의미 입력이 남는다. provenance SHA는 branch 표현을 대신하지 않으며 미관측 rank를 만들어 채우지 않는다. 제품 V-free 조건은 유지한다. |
| 후속 label chain(DG-05) | 선행 label 존재·동일 immutable input 귀속·causal revision·cycle/missing predecessor 거부·현재 학습 view 선택 감사가 남는다. canonical observation revision과 dataset label revision은 다른 소비 경계다. |
| 일반 dataset frozen identity(DG-06) | 등록된 game·producer별 OwnPals source·model·frozen epoch의 일관성을 일반 admission에서도 검사하는 계약이 필요하다. 단일 native collector의 보장을 임의 등록 dataset의 보장으로 확대하지 않는다. 서로 다른 두 엔진의 모델이 다를 수 있으므로, 선언된 engine roster를 구분하지 않고 game 전체에 하나의 모델을 강제하지 않는다. |
| 양의 owned target 자료(DG-07) | positive fixture loss와 실제 owned 양의 target 자료를 구분한다. 기존 6행의 all-masked 전체 소비와 V의 unknown mask는 올바른 미관측 처리이며 학습 개선이나 API 미구현의 대체 증거가 아니다. |
| 수정된 최종 종단 인수 | 새 검사·정확한 source/binary 등록 뒤 유한 typed/Core 저장·시계·PGN·실패 보존을 따로 인수한다. CPU06의 runner 정리 성공이나 과거 Core projection fixture를 새 live-owner 종단 성공으로 바꾸지 않는다. |

후속 담당의 source 확인에 따라 C divergence의 미완료 범위를 위 연결부로 명시한다.
기존 감사의 미완료 판정을 **native capture 전체가 없다는 뜻으로 확대하지 않는다.**
기존 감사 원문은 보존하며, 실제 capture·학습 입장 보류·counterfactual WDL mask와 남은
감독·loader·ranking 준비를 이 정정 기록에서 구분한다.

제품 GPU startup·독립 수치·취소·물리 drain·paired는 계속 `user-deferred`다. 이 문서
갱신은 새 GPU 실행·자동 재시도 권한을 추가하지 않는다. 원문 후속 모델 구조·압축·
warm-start·스케줄 후보도 보존하며, PALS 전체 도입에는 의미 보존 변경의 5% 문턱을
적용하지 않는다. 실제 학습과 학습된 기력 검증은 이번 목표의 제외 범위로 유지한다.

### 후속 근거 식별

작은 JSON·감사 문서의 아래 식별을 대조했다. 원시 PGN의 식별은 CPU06 failure review가
연결한 보존 자산이며 이 절에서 PGN을 재작성하거나 새 경기 결과로 변환하지 않았다.
원시 로그·개인 경로·호스트 정보는 소스 문서에 옮기지 않는다.

| 관리 루트 기준 논리 자료 | SHA-256 |
|---|---|
| `reports/pals/completion-audit-30f2a0d-02.md` | `3c827e6c48e2d83579df53a1ca2c76755518f88d0ce7fa076ebad6a19b3aea8b` |
| `runs/pals/own-paired-cpu-06/failure-review-02.json` | `9c196441d84efc080241c7de5589a3000581613089237dae2e2b6a60dc4dd799` |
| `runs/pals/own-paired-cpu-06/execution-result.json` | `1a2f9c2cfa11a16b6f4e4746836ad2038b2b4da24945f77709a50361da77dd77` |
| `runs/pals/own-paired-cpu-06/recovered/attempt-01/match.pgn` | `7b407a18753646b4517eb6ef2c972f6ca52929cc3f6c90bd23b88b531ae9674b` |
| `runs/pals/model-reference-10/validation.json` | `98b7f745e7a83facbbc2dbc384c093241608022f403e53dbc91c4df8c0ba263b` |

## 2026-10-07 후속 진행 — 현재 라벨과 외부 checker의 탐색 소비

이 절은 앞선 실패·인수 기록을 보존한 추가 상태다. GPU는 사용자 지시에 따라 이번
세션에서 보류한다. 실제 optimizer 학습은 여전히 제외하며 CPU 검사의 성공을 GPU·
학습·기력 인수로 바꾸지 않는다. PR #23은 `develop` 대상 Draft로 유지한다.

### 라벨 계보와 실제 소비자

`9c91128`은 선행 라벨의 존재, 같은 immutable input 귀속, 엄격히 증가하는 관측
sequence와 단일 causal chain을 검증한다. fork·duplicate·missing predecessor·cycle·
다중 labeled root를 거부하고 원시 행·raw digest·split은 보존한다. 최신 whole label만
현재 학습 view에 들어간다. 최신 라벨이 policy/value를 mask하면 이전 target을 자동
병합하거나 되살리지 않는다. unlabeled capture는 labeled chain의 별도 root가 아니다.

Rust/Python의 새 `rz-pals-label/1`과 current view는 확률의 f64 bit 표현을 정규화하여
같은 식별을 계산한다. 기존 input·raw dataset·split·checkpoint domain은 바꾸지 않는다.
Private V context는 공개 label seal 밖에서 별도 실제 query·control 검증을 계속한다.
현재 leaf만 읽어 원시 이력의 변경을 생략하는 입장은 허용하지 않는다.

`5ee8f1c`는 준비 프로그램과 V producer에 이 current view를 연결한다. 모든 원시
자료의 검증을 유지하면서 현재 input을 한 번만 선택하며, V의 `max_steps`는 raw row
index가 아닌 선택된 input의 순번에 적용한다. 준비 영수증은 raw count·raw hash와
current count·view hash를 따로 기록한다. 기존 읽기 경로를 실제 학습 실행으로 표시하지
않는다. 총괄의 Rust data 검사 25개와 후속 Python 전체 72개가 통과했다. Python 검사는
CPU fixture·준비·소비 경계이며 optimizer update와 GPU 실행은 0이다.

엄격한 계보 검사를 연결한 뒤 실제 collection producer의 결함도 드러났다. 기존
OwnedCpu policy와 후속 ActualGame 결과가 같은 input에 각각 predecessor 없는 labeled
root를 만들었다. 최신 `5ee8f1c` CI의 Linux/Windows 수집기 4개 실패도 이 경계를
확인했다. CPU bindings와 model CPU job의 성공은 이 실패와 구분한다. 수정 중인
producer는 실제 append에 성공한 선행 라벨의 digest만 연결하며 저장 실패 후 rows·
predecessor index를 진행시키지 않는다. 원래 실패 로그는 보존한다. 이 수정의 최종
검사·커밋·CI는 별도 후속 증거로 기록하며 이 절에서 성공으로 미리 선언하지 않는다.

### 외부 checker와 모델 WDL의 탐색 접점

`86affc1`은 search의 checker 소비와 Native 모델 값 접점을 공유한 WIP다. 외부 보고서의
cp·mate·bound·reported depth·선택적 탐색·미관측 작업량은 자체 CPU raw score나
완료 iteration, WDL 또는 Rules 증명으로 변환하지 않는다. raw 보고서와 실제 관측
작업량, 예약한 node 예산을 독립 보존한다. unknown nodes는 unknown으로 남긴다.

외부 PV를 정확한 Rules 상태·이력으로 재생한 뒤 해당 후보의 frontier를 선택한 모델의
기존 contextual WDL로 다시 평가한다. `ModelValueIdentity`와 실제 prepared input key를
검증하고 W/L 관점 반전과 draw 보존을 적용한다. 미검사 방어를 전체 게임의 승패로
확정하지 않는다. 직접 현재 상태의 Rules terminal과 제한된 경로의 terminal 전파는
`RulesTerminal`·`RestrictedRulesLine`으로 구분한다. 별도 resolver는 CP 보정·평균·
unknown의 0점 대체를 수행하지 않는다.

Native 값 조회는 기존 Proposer forward의 shared WDL을 사용한다. 새로운 candidate
value head·인코딩을 도입하지 않는다. 물리 실행 전에 만든 input key를 실제 반환에서
전달하고, 조회 완료와 탐색 소비를 구분한다. 뒤늦게 남은 시간 특징을 다시 해시하여
준비 입력을 바꾸지 않는다. 기존 단일 물리 worker·buffer 소유권·finish fence를 유지한다.

총괄의 마지막 중앙 library 검사에서 search 163개·UCI 135개가 통과했고 두 패키지의
all-target/all-feature Clippy `-D warnings`도 통과했다. 앞선 컴파일 오류와 fixture 실패를
보존했다. 검사 수는 각각 해당 실행의 수이며 과거 겹치는 검사를 합쳐 고유 검사 수나
현재 통합 전체 성공으로 표시하지 않는다.

외부 helper의 제품 CLI·등록 profile·두 owner의 독립 종료·receipt·실제 arena 인수는
후속 연결이다. `86affc1`의 search 소비 성공이 이 경계의 완료를 뜻하지 않는다. 공개
foreign WDL 결론 특징, record별 native K/V 소비, 역할별 warm-start, DG01~04·DG06~07의
남은 학습 준비 계약도 위 전체 목표 감사와 함께 유지한다.

### 후속 중앙 인수 — producer 수정과 제품 checker 수명

`0ee4669`는 실제 collection producer가 같은 input의 후속 label을 저장할 때 마지막
성공 append의 digest를 predecessor로 연결한다. 서로 다른 label owner를 독립 유지하며,
실패한 append 뒤 rows·causal index를 진행시키지 않는다. 중앙 collection 검사는 순차
21개와 네 test thread의 21개가 각각 통과했다. `5ee8f1c`의 네 CI 실패는 보존하며,
`0ee4669`의 CI run `37631162901`에서 Linux·Windows·CPU bindings·model CPU 네 job이
모두 성공한 것을 확인했다. 이 결과는 후속 dirty 소스나 다른 SHA의 CI 성공이 아니다.

`af775ad`는 외부 helper의 등록·실제 startup·게임 초기화·유한 종료를 제품 driver와
CLI에 연결한다. 기본 own 경로의 v1 identity를 보존하고 명시적 external checker에는
별도 resolver·등록 digest를 사용한다. helper는 unstarted 상태로 구성하며, 모델 준비와
driver 구성 이후 유한 시계·취소 아래 실제 시작한다. 준비 실패·영수증 저장 실패·대국
서비스 실패에서는 Native와 helper의 종료를 각각 시도하고 각 원인을 보존한다.

제품 선택은 `--pals-cpu-checker=external-uci`, `--pals-cpu-profile`,
`--pals-cpu-profile-sha256`로 명시한다. 첫 profile은 내장 NNUE Stockfish의 제한된
등록 형식이며 파일 SHA·canonical SHA·지원 옵션 검사와 실제 UCI 식별을 구분한다.
`readyok`나 설정 송신만으로 적용값·학습 이력·모델 로딩을 관측했다고 표시하지 않는다.
contextual 모델 WDL을 제공하지 않는 mock에 외부 checker를 조용히 연결하지 않는다.

V3 Native 영수증의 선택적 `cpu_checker`는 두 owner의 근거를 분리한다. 시작 전
실제 cleanup과 시작된 process의 exit·pipe drain을 구별하고, 미관측 값은 null/unknown으로
남긴다. work 관측이 실패하면 서비스 종료를 성공으로 게시하지 않는다. 외부 cp·mate·
reported bound가 Native 물리 완료나 Rules 증명을 대신하지 않는다.

소스 pin을 고정한 `rz-search`·`rz-uci` all-target/all-feature 중앙 검사 468개와 Clippy
`-D warnings`가 통과했다. 앞선 API 접점의 컴파일 실패 로그를 보존했다. 관리 build
slot은 이 검사 종료 후 정책 상한을 넘은 재생성 산출물만 회수했으며 소스·인수 자료는
보존했다. 이 회수는 테스트 성공이나 메모리 성능 개선의 대체 근거가 아니다.

`969099e`는 arena의 기존 own V3 소비자가 새 work completeness boolean을 잘못
u64로 읽는 경계를 수정한다. 해당 필드만 PALS의 bool/null로 읽고 startup의 true·
CPU 경로 혼입·잘못된 타입은 거부한다. 기존 mandatory numerical counter 조건은
유지한다. 중앙 `pals_launch` 검사 24개가 통과했다.

위 제품 단위의 CPU 성공과 실제 외부 helper·arena 인수는 구분한다. 기존 own V3의
raw resolver·자원·Core counter 조건은 외부 profile에 그대로 사용할 수 없다. 명시적
외부-helper launch 등록, helper 자산 snapshot·합산 자원·종료 영수증, 실제 Stockfish
실행과 수정 이후 paired 종단 인수는 여전히 후속 작업이다. DG06 metadata 구현도
live producer·loader·resume 입장과 구분하여 별도 중앙 검사로 인수한다.

### DG06 metadata와 실제 helper 식별의 추가 단위

`44f65e4`는 Linux helper의 spawn 뒤 실제 `/proc` PID·process group·start tick 대조에
성공한 경우에만 `ExternalProcessIdentity`를 보존한다. 종료 후 같은 역사 식별을 유지하며
own checker·지원하지 않는 호스트·spawn 이전 실패에는 None을 남긴다. nested checker
영수증에 이 식별을 추가했으며 exit·pipe drain·Native NN 완료와 독립적이다. 중앙 search
164개·UCI 152개 library 검사와 두 패키지 all-target/all-feature Clippy가 통과했다.
실제 arena의 inherited cgroup·helper PID join 인수는 이 타입 추가만으로 완료되지 않는다.

`17fc6c5`는 DG06의 첫 metadata 계약이다. 별도 domain의 producer roster·unique input
capture binding·envelope가 기존 raw dataset·split·current view와 독립 등록 pin을 묶는다.
같은 게임의 두 producer는 다른 모델을 가질 수 있지만 각 producer의 source·epoch·
등록 encoding은 모든 원시 이력에서 고정한다. 기존 input/raw/split/current seal과 label
계보를 다시 검사하며 label provenance나 차례에서 input owner를 추정하지 않는다.

NativeExact는 실제 선언된 encoding을 대조한다. PrivateCheckedDerivedQuery는 별도
query schema·encoder·parent pin을 보존하고 항상 `requires_derived_adapter`에 남긴다.
schema SHA의 형식 일치만으로 실제 private query·tensor 검사를 통과시키지 않는다.
Python metadata 경계도 independently pinned raw receipt와 actual records/registry bytes,
별도 fully validated current-view pin을 요구하며 기존 raw f64 seal을 새 표현으로 바꾸지 않는다.

소스 pin을 고정한 중앙 Rust `pals_data` 33개(기존 25개+신규 8개), stdlib Python metadata
12개, `rz-experiments`·`rz-arena` all-target/all-feature Clippy가 통과했다. Python의 이번
검사는 모델·Torch·optimizer를 실행하지 않는다. 결과 scope는 항상 `metadata_only`이며
실제 producer 사용·학습 입장·전체 목표 완료를 뜻하지 않는다.

다음 연결은 독립 등록을 검증한 producer handle, 실제 capture evidence, 기존 출력
budget 안의 roster/capture/envelope 영수증과 strict loader/resume이다. Native capture의
현재 의미는 seal-before-submit·prepaid drain-after-search이며 durable disk write-before-submit
증거로 보고하지 않는다. prepared journal 전체와 raw learning history의 unique input binding을
구분해 divergence·거절·실패 입력을 보존한다. 기존 legacy loader/checkpoint에는 새 pin을
자동 생성하여 strict 인수로 승격하지 않는다.

### 외부 CPU_R 선택과 host record projection page의 첫 연결

`3e2125b`는 명시적 `cpu_r` 선택을 V3 명세·lock에 보존한다. 기존 Own 선택의
직렬화 bytes와 canonical digest는 유지하며, 외부 profile·binary·모델 WDL resolver·
자원 및 유한 수명 선언을 별도 타입으로 검증한다. 중앙 manifest library 검사 21개가
통과했다. 현재 실제 arena launcher·helper cgroup·Core projection 인수는 연결 전이므로
외부 선택의 실행 검증은 거부한다. 선언을 읽었다는 사실을 실제 자원 적용으로 표시하지 않는다.

명시적 `enable_host_record_pages`는 기존 whole-input 캐시와 별도의 물리 projection
page 경로다. 실제 16개 FP32 feature bit와 public graph·checkpoint·encoding·game을
식별에 묶고, 전체 canonical input·관측·CPU task identity는 그대로 보존한다. missing
record를 하나의 subset Run으로 공급하고 원래 head-major 순서의 K/V와 mask를 join하여
기존 private P/C Run에 전달한다. 빈 record의 false-mask padding도 실제 zero-feature
projection을 사용한다. 같은 feature의 물리 page 공유가 관측·방문·CPU 검사 재사용을 뜻하지 않는다.

페이지 budget은 entry backing과 실제 소유 배열을 포함하며 transient reservation은
준비 입력·subset·join·출력·page 복사의 겹치는 수명을 포함한다. known completion에서만
pin과 scratch를 해제하고 unknown physical completion에서는 session·full input·subset·
page·pin·unfinished join을 같은 owner에 보존한다. ORT workspace·allocator overhead·
프로세스 peak·VRAM peak는 이 호스트 예약으로 관측했다고 주장하지 않는다.

동결한 4개 소스의 중앙 `rz-eval` library 76개와 `rz-runtime` library 29개, 두 패키지
all-feature library Clippy `-D warnings`가 통과했다. actual CPU whole/page 수치 동등성은
별도 검증 전이며 제품 CLI·Native receipt 선택은 아직 연결하지 않았다. subset Run에서도
기존 public graph는 board 66개 토큰을 다시 계산한다. page hit·encoded record slots·
physical B1 NN 입력·탐색 소비 수를 구분하며 속도·메모리 개선을 주장하지 않는다.

`record-pages-numeric-01`은 `c6ead34`와 별도 예제 source pin의 실제 CPU ONNX
정확성 인수다. 기존 PyTorch reference 6개의 whole 결과를 유지하며 page 경로도 같은
6개 reference에 대조했다. 추가 whole/page 파생 입력 15개는 0/1/2/128 records,
padding과 실제 zero-feature record, append·correction·reorder·critical·ID 이동·삭제·
P/C 순서를 포함한다. 최대 policy 절대 차이 `5.96046448e-8`, WDL `8.94069672e-8`,
K/V key `1.19209290e-6`, private latent `1.66893005e-6`는 기존 허용 오차를 만족했다.
mask는 정확히 일치했다. 파생 입력은 모델 tensor 수치 검사이며 Rules 인증·CPU 관측·
학습 목표의 증거가 아니다.

반복·reorder·critical·ID 변경에서는 public Run이 늘지 않았고, 단일 append·correction은
missing record 한 슬롯만 공급했다. clear는 NN 실행 없이 page와 witness를 비웠으며
새 게임 이후에는 실제 public Run을 확인했다. 완료 뒤 pin·subset·join scratch·prepared
input·transient reservation이 해제됐고 프로세스 exit 0과 관리 supervisor의 자식 정리도
확인했다. 예제 all-feature Clippy가 통과했다. 실제 학습은 0 step이며 GPU는 user-deferred다.
