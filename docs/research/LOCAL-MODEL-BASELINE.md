# 로컬 LC0 실행과 강도용 모델 전환 계획

기준일: 2026-10-04. 총괄 TASK-I01/I02의 조사·계획과 후속 구현 기록이다. 사용자는 현재 Maia의
강도 한계를 검토하고 LC0 로컬 실행 가능성을 평가한 뒤 모델 교체 또는 추가 학습
계획을 요청했고, 후속으로 **BT4-it332의 실제 적용과 LC0/RoveZero 벤치마크**를 지정했다.
초기 T1 조사는 보존하며 현재 구현 대상은 BT4로 변경한다. 학습·정식 강도 인수는
수치 대조·탐색 벤치마크와 구분한다.
기준 소스는 `develop`의 `87017a27a467dd37ca711a39020f3ddf5d6a9934`다.

## 1. 현재 기준선의 한계와 확인된 사실

- 최초 `develop` 기준 Rust evaluator는 Maia1 v1.0 `maia-1900`의 6개 SE residual block·64 channel
  CNN이다. 인간의 수를 예측하는 호환·런타임 기준선으로 선정했다. 이름의 1900은
  현재 RoveZero의 Elo나 엔진 강도 보장이 아니다.
- CNN이라는 구조만으로 결함을 판정하지 않는다. 학습 목표·가중치 품질, 입력/
  policy/WDL 의미, 탐색 예산과 runtime 수명은 각각 대조한다. Maia 공식 실행 예는
  인간 수 재현을 위해 `go nodes 1`을 사용한다. 강한 대국 엔진과 목표가 다르다.
- 기존 48판은 같은 Maia·같은 PUCT의 pending/batch 폭 1/4 내부 비교였고 득점률은
  50%였다. 강한 모델이나 LC0와의 대국 근거가 아니다. WorkerLimit/legal fallback
  3건과 bestmove 뒤 physical GPU 완료·소비 journal의 공백을 보존한다.
- 최초 B의 기본값과 설정 검증에는 **128 simulation 상한**이 있었다. 후속 BT4 연결에서
  기본값 128은 보존하고 native 실행에 `--search-simulations=1..4096`을 추가했다.
  예산 확장과 시간 제어·수명 문제의 해결은 별도다. LC0의 UCI nodes와
  RoveZero simulation을 같은 비용·같은 방문으로 취급하지 않는다.

근거: [Maia 공식 설명](https://github.com/CSSLab/maia-chess),
[첫 선정](../WEIGHT-SELECTION.md), [기존 내부 A/B](LOCAL-STRENGTH-AB.md),
`crates/rz-eval/src/asset.rs`, `crates/rz-uci/src/engine.rs`.

## 2. 실제 로컬 LC0 실행

공식 stable LC0 v0.32.1의 Windows CUDA 12 package를 사용했다. 유료 GPU·RunPod를
사용하지 않았다. package의 공개 SHA-256과 다운로드 bytes를 대조했다.

| 항목 | 잠근 조건 / 관측 |
|---|---|
| 장비 | RTX 4050 Laptop 6 GB, driver 610.62, Windows; RAM 약 15.6 GiB |
| 엔진 | LC0 v0.32.1, classic search, 공식 Windows CUDA 12 package |
| package SHA-256 | `8d0ce17676eb15e303bea9e790742d31c94ce5d24107f6187adc58e820f6d2f7` |
| executable SHA-256 | `e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b` |
| backend | 명시적 `cuda-fp16`; UCI option과 CUDA 12.9 / 해당 GPU runtime 진단 확인 |
| 자원·옵션 | threads 1, minibatch 16, NNCache 10,000, cache history 7, Ponder off |
| 시간 | MoveOverheadMs 10, SmartPruningFactor 0; startup와 착수 시계 구분 |
| 입력 | 시작 상태, 흑 차례 오프닝, Sicilian 이력, pawn endgame, 흑/백 mate-in-one의 6개 상태 |
| 요청 | 모델별 각 상태의 `go nodes 128`, `go movetime 100`, `go movetime 1000`: 18회 |
| 모델 | Maia 원본과 T1-256x10-distilled-swa-2432500 원본을 별도 process로 순차 실행 |
| 완료 | 두 모델 모두 18회 bestmove와 exit 0, process 종료 확인; 독립 oracle로 36개 착수와 전체 PV의 합법성·printed WDL 범위/정규화 감사 |
| GPU 메모리 | 두 모델을 순차 실행한 구간의 전체 장치 관측 최고 1,169 MiB; process peak가 아님 |
| 상한 | process별 120초, 응답별 15초, log stream별 8 MiB, model 50 MiB 다운로드 상한 |

`python-chess` 1.11.2로 36개 입력/착수/PV를 감사했고 흑·백 mate-in-one 12개 착수는
실제 checkmate를 확인했다. 100 ms 요청의 bestmove 응답은 Maia 90.30~91.18 ms,
T1 90.31~93.62 ms였고 1,000 ms 요청은 각각 990.26~991.22 / 990.29~992.38 ms였다.
각 모델 6개의 단일 smoke 표본이며 속도 분포·전체 강도·per-move physical GPU drain을
입증하지 않는다. 이 smoke는 실제 GPU 실행 가능성 근거이며 FP32/FP16 수치 parity·
동시 두 엔진 VRAM·정식 GPU 시간 공정성의 인수가 아니다. raw receipt/log는 저장소 밖
`reports/coordinator-integration/lc0-model-evaluation-20261004/`에 보존한다.

첫 Windows `--help` 호출은 20초 timeout, 재시도는 exit 0이었다. 원인은 미확정이다.
또 최초 time smoke의 LC0 기본 MoveOverheadMs는 **200 ms**여서 100 ms 요청은 nodes 1로
종료됐다. 원본 기록을 보존하고 `probe-v2`에서 위 10 ms·pruning 0·Windows QPC
단조 시계로 다시 실행했다. 기본 옵션으로 상대의 탐색 예산을 약화하지 않는다.

재현 argv는 별도 빈 config와 명시 weights 경로에 아래 옵션을 합친다.

```text
lc0 --config=<empty-config> --weights=<exact-weight-file> --backend=cuda-fp16
    --threads=1 --minibatch-size=16 --nncache=10000 --cache-history-length=7
    --show-wdl --preload --move-overhead=10 --smart-pruning-factor=0
```

[공식 release](https://github.com/LeelaChessZero/lc0/releases/tag/v0.32.1),
[공식 모델 목록](https://lczero.org/play/networks/bestnets/).

### T1 ONNX 변환과 외부 CPU 수치 대조

같은 공식 LC0 converter로 `leela2onnx --onnx-data-type=f32 --onnx-opset=17
--onnx-batch-size=-1`을 실행해 **80,896,290 bytes**의 단일 ONNX를 만들었다.
SHA-256은 `0e2699c19b617781d4fbe3d00286756b125836a30c0658afc620e197bec35d04`다.
ONNX 1.18.0 checker를 통과했고 external tensor는 없다. graph는 node 555개,
initializer 377개이며 다음 인터페이스를 직접 확인했다.

| FP32 인터페이스 | shape / 출력 의미의 확인 범위 |
|---|---|
| `/input/planes` | `[batch,112,8,8]` |
| `/output/policy` | `[batch,1858]`, 마지막 op Gather, raw logits로 원본 참조와 대조 |
| `/output/wdl` | `[batch,3]`, 마지막 op Softmax, W/D/L 확률로 원본 참조와 대조 |
| `/output/mlh` | `[batch,1]`, 마지막 op Relu; NN moves-left 출력이며 규칙 판정을 대신하지 않음 |

고정 LC0 `fd71a2d921b689c5f479d3227c3806c8e272d9c5`의 Eigen backend/원본
protobuf와 ONNX Runtime 1.22.0 CPU/FP32를 같은 LC0 입력으로 대조했다. thread 1,
NumPy 2.2.6, 같은 6개 상태에서 실행 전 고정한 absolute tolerance는 raw logits
`5e-4`, WDL와 합법 policy `1e-5`, batch/single `5e-4`다. 최댓값은 각각
**1.4306e-5 / 3.1293e-7 / 8.5140e-7** 이하여서 통과했다. batch 2/4/8/16과
single의 출력 차이 최댓값은 0이었다. Eigen module SHA-256은
`c795ba5809490beaf3117b4d94bb42616af4f0aaee22e2e02012144356f054f4`다.

이는 외부 CPU reference/export 검사다. 입력 생성도 LC0를 사용했으므로 **현재 Rust
encoder의 T1 parity, Rust loader, ONNX CUDA·FP16 parity는 아직 인수하지 않았다.**
raw graph/CPU reference receipt는 같은 외부 루트의 `t1-export/`에 보존한다.

## 3. 초기 소형 후보 조사: 동결 T1 distilled

**추천 순서는 동결된 강한 사전학습 모델의 연결·평가를 먼저 하고, 그 뒤 학습이다.**
초기 추천 후보는 공식 목록의 소형 T1-256x10-distilled-swa-2432500이었다. 외부 LC0의
실행 가능성이 확인됐고, 내부 Rust 모델 승격은 아직 하지 않았다.

후속 사용자 지정으로 실제 연결·벤치마크 대상은 **BT4-it332**다. T1의 제작자 허가는
BT4에 전파하지 않는다. 이번 BT4는 공식 권장 다운로드의 로컬 연구 실행이며 개별
가중치 license는 미확인으로 보존한다. 원본·변환 가중치를 Git/배포물에 넣지 않는다.

| 원본 필드 | 직접 검사한 값 |
|---|---|
| 공식 URL | `https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz` |
| gzip bytes / SHA-256 | 37,118,673 / `bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406` |
| protobuf bytes / SHA-256 | 40,401,217 / `6ad1b1ceae674911a7fe5a0b3369d0bc5b1f68e655fbc8b302ebb4e9feb34bc0` |
| 최소 LC0 / 저장 형식 | 0.29.0 / LINEAR16; 실행 FP16과 저장 양자화는 별개 |
| 입력 | INPUT_CLASSICAL_112_PLANE = 1 |
| body 메타데이터 | NETWORK_SE_WITH_HEADFORMAT = 4; residual 0, attention encoder 10, headcount 8, embedding 256 |
| policy / value | POLICY_ATTENTION = 3 / VALUE_WDL = 2, OUTPUT_WDL = 2 |
| moves left / activation | MOVES_LEFT_V1 = 1 / DEFAULT_ACTIVATION_RELU = 1 |
| 내장 license | 없음; 아래 직접 제작자 답변을 별도 보존 |

body enum 이름만 보고 SE CNN이라고 해석하지 않는다. 실제 encoder와 tensor 구조를
함께 검사한다. 입력 enum이 Maia와 같아도 history fill·관점·승격·캐슬링 mapping을
자동 호환으로 인정하지 않는다. LC0 명세의 고정 참조 SHA는 기존 선정 문서와 같다.

2026-08-05 제작자 `masterkni6`는 해당 파일을 제작했다고 확인하고 누구나 원하는
방식으로 사용할 수 있다고 답했다. 이번 로컬 사용·후속 호환 연구는 그 직접 허가를
근거로 삼는다. 이를 임의의 MIT/CC0/SPDX license로 바꾸지 않으며 출처·원본 digest와
답변을 보존한다. 실제 재배포 산출물·modified checkpoint의 고지 조건은 별도 기록한다.
[제작자 답변](https://github.com/orgs/LeelaChessZero/discussions/2430)

공식 표의 약 1.6 GB는 안내값이다. 이번 장치 전체 표본의 1,169 MiB와 정의가 다르다.
이 초기 T1 조사에서는 BT4를 실행하지 않았다. 후속 사용자 지정에 따라 아래 BT4를
실제로 적용했다. 공식 약 4 GB 안내만으로 동시 점유·속도·권리·학습 적합성을 판단하지 않는다.

## 4. BT4-it332 실제 Rust 연결과 수치 인수

2026-10-04 사용자가 BT4-it332 적용과 native LC0/RoveZero 벤치마크를 지정했다.
RoveZero의 자체 Rules·최소 PUCT·runtime은 그대로 사용하고 평가 모델만 BT4로 선택했다.
LC0는 원본 수치 참조·converter·외부 상대이며 RoveZero의 탐색을 대신 실행하지 않는다.

| 잠근 식별자 | 값 |
|---|---|
| 공식 원본 | [BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz](https://storage.lczero.org/files/networks-contrib/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz) |
| gzip bytes / SHA-256 | 382,645,315 / `e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618` |
| protobuf bytes / SHA-256 | 382,616,086 / `d6e4bbf289bea1fe312b7a0286106aeb713760b604c932ef8cdeebf16a23f36c` |
| 형식 | 최소 LC0 0.30.0, LINEAR16 저장, classical 112-plane, attention body/policy, WDL, MLH |
| body | encoder 15·head 32·embedding 1024·FFN 1536, dense positional embedding |
| 변환 | 고정 LC0 `fd71a2d921b689c5f479d3227c3806c8e272d9c5`, `leela2onnx`, FP32·opset 17·policy vanilla·value winner |
| ONNX bytes / SHA-256 | 741,143,425 / `2839171c39fe660ca5fa35983bba7d0b403bc6e70b56a06b88a06b406d057518` |
| graph | node 822·initializer 553·external tensor 0, 동적 batch input `[B,112,8,8]`, policy `[B,1858]`, WDL `[B,3]`, MLH `[B,1]` |
| export manifest SHA-256 | `9aeed58b0c8a3cc5ab7a684ae9e5de567028d79f9a0ccf0c81f6fd0df861e279` |
| 권리 | 개별 weights license 미확인, `UNVERIFIED-local-research-only`, `redistribution_ready=false`; 원본·변환본은 외부 보존 |

`AssetProfile::Bt4It332`은 두 원본 digest가 함께 일치해야 선택된다. Maia의 exact
source·16 MiB ONNX 상한과 기존 license 검사를 보존하고 BT4만 768 MiB 상한·세 출력으로
분리했다. `MaiaAsset`는 `SelectedAsset`의 호환 별칭이며 모델 ID/cache namespace는
manifest identity로 구분한다. BT4 MLH는 검증한 graph에 존재하지만 RoveZero 탐색은
소비하지 않는다. native LC0와의 차이를 단일 PUCT 효과로 해석하지 않는다.

큰 원본 검증은 64 KiB 버퍼로 gzip을 스트리밍하며 protobuf 전체 복제를 피한다.
CUDA arena/session 선언은 Maia 1 GiB·BT4 3 GiB로 C/D/native receipt에 함께 적용했다.
이 선언은 **전체 VRAM 실측·외부 hard cap이 아니다.** CUDA/FP32·TF32 off·CPU fallback
금지와 실제 node placement·19-file bundle 검증을 유지한다. 공통 계약 revision은 0.1이다.

기존 native CLI에 외부 산출물을 지정해 재현한다. 아래 경로는 실행 환경의 저장소 밖
실제 파일로 바꾸며 각 manifest/runtime/bundle SHA를 확인한다. CPU는 `--onnx-cpu`와
고정 CPU runtime을 별도 선택한다. GPU profile을 조용히 CPU로 바꾸지 않는다.

```text
rz-uci --onnx-cuda --source-weights=<BT4.pb.gz> --onnx-model=<BT4-fp32.onnx>
       --export-manifest=<manifest.json> --manifest-sha256=<exact-manifest-SHA>
       --ort-library=<ORT-CUDA-core> --ort-sha256=<exact-core-SHA>
       --cuda-bundle=<nineteen-file-spec.json> --cuda-bundle-sha256=<exact-spec-SHA>
       --output-root=<new-external-run-directory> --search-simulations=4096 --attestation
```

UCI에서는 `uci`/`isready` 후 완전한 `position startpos moves ...`와 유한 `go movetime`
또는 `go nodes`를 전달한다. 벤치마크는 pinned original/ONNX/reference와 동일한
immutable binary로 실행하며 최신 문서 head의 identity와 실제 실행 source를 구분한다.


C/A/D 수치 인수 소스는 `aa0cc0247b9d7041e4525b2021363d0617cfd0bc`이며 고정 12개
독립 LC0 Eigen 원본 참조를 사용했다. 양쪽 차례·충분/짧은 이력·반복·같은 보드와
다른 이력·EP·양쪽 캐슬링·네 승격을 포함한다. 외부 ONNX CPU의 허용 오차는 실행 전
raw logit `5e-4`, WDL·합법 policy `1e-4`, batch/single `5e-4`로 잠갔다. 관측 최대는
각각 **6.4373e-5 / 4.7684e-7 / 4.5773e-6**이고 batch 2/4/8/16 차이는 0이었다.

Rust C raw CPU/CUDA의 12개 참조·batch 1/2/4/8/16·worker 검사를 통과했다. Rust CUDA
raw 검사에는 GPU에서 실행한 node 687개가 기록됐으며 batch/single 차이 최대는
`1.9253e-5`다. 실제 A immutable 상태→C→D CPU/CUDA 검사도 각각 12개(No 5·Repeat 7)
통과했고 physical drain·잔여 예약 0을 확인했다. 이는 수치·해당 수명 경로의 인수이며
모든 취소 race·전체 UCI 시계 공정성·강도 인수가 아니다. CUDA 원본 수치 검사의
raw logit 허용값은 `atol=1e-4, rtol=1e-3`, WDL·합법 policy는 `1e-4`다.

## 5. 동일 BT4의 로컬 위치 벤치마크

실행 소스는 `78b7c53502cadaff77fc6de5f0832eee55b9938c`, RoveZero binary SHA-256은
`1a4b976b4110d251e21b09ed1e1f29b398b4ed01f533788a941de8dc4be969b3`다.
6개 상태 × `go nodes 128`/`go movetime 1000`/`go movetime 5000`을 각 profile에서
실행해 **54개 합법 착수·세 process exit 0**을 확인했다. 각 상태/조건은 단일 표본이다.

| profile | 실행 조건 | 전체 장치 관측 최대 VRAM | 준비→ready |
|---|---|---:|---:|
| LC0 FP32 | Windows v0.32.1 `cuda`, max_batch 256/min_batch 4, minibatch 16 | 2,009 MiB | 1.79초 |
| RoveZero FP32 | Ubuntu WSL2 ORT 1.22 CUDA, TF32 off, B1, 4096 simulation cap | 1,131 MiB | 36.41초 |
| LC0 FP16 | 별도 정밀도 profile `cuda-fp16`, 나머지 LC0 옵션 동일 | 1,083 MiB | 2.67초 |

VRAM은 250 ms 간격 `nvidia-smi`의 **전체 장치 관측값**이다. 짧은 peak·process별 peak·
모든 batch 크기의 요구량을 보장하지 않는다. 공식 약 4 GB는 조건이 고정된 실측값이
아니며 이번 제한된 batch에서는 6 GB 장치에 실행할 수 있었다. FP16 LC0를 FP32
RoveZero와 같은 정밀도의 알고리즘 성능 비교로 사용하지 않는다.

| 일반 opening 세 상태의 중앙 bestmove 응답 | LC0 FP32 | RoveZero FP32 | LC0 FP16 |
|---|---:|---:|---:|
| `nodes 128` 요청 | 1,051.10 ms | 1,679.71 ms | 272.90 ms |
| `movetime 1000` 요청 | 991.81 ms | 1,012.76 ms | 992.15 ms |
| `movetime 5000` 요청 | 4,998.02 ms | 5,058.68 ms | 4,997.90 ms |

일반 opening은 start/Ruy-black/Sicilian이며 각각 한 번 실행했다. LC0의 실제 nodes는
요청값보다 많을 수 있고 RoveZero simulation과 동일한 단위가 아니다. RoveZero는
이번 UCI에서 `info nodes/nps/pv`를 제공하지 않아 NPS/PV 비교는 **미측정**이다.
LC0의 36개 printed PV만 독립 oracle로 검증했다. raw 초기 auditor의 PV bool은
미제공 PV도 true로 표시했으므로 별도 `positions/analysis.json`에 존재 여부와 null을
명시했다. raw 결과는 보존하며 미제공 PV 검증을 주장하지 않는다.

양쪽 차례 mate-in-one × 세 요청에서 LC0 두 profile은 각각 **6/6**, RoveZero는
**0/6**의 즉시 메이트를 선택했다. RoveZero는 백 `g6f6`, 흑 `g3f3`을 선택했고 모두
합법이다. 현재 root 선택은 방문 수 우선이며 확정 메이트 우선 처리나 MCTS solver가
없다. 이 관측만으로 가치 부호 오류나 BT4 인코딩 실패라고 단정하지 않는다. 강한
weights를 넣어도 최소 탐색이 LC0 수준이 되지는 않는다는 구체적인 후속 검증 대상이다.

RoveZero 일반 상태 5초 응답의 최댓값은 5,295.06 ms다. Expired/Stale 진단을 보존했고
WorkerLimit/LegalFallback 문구는 없었다. 문구 개수는 고유 요청 실패 수가 아니다.
자연 완료 경로는 runtime shutdown을 완료한 뒤 Event::Complete를 게시하도록 보완하고
일부러 shutdown을 지연하는 회귀 검사를 통과했다. 독립 hard deadline/stop의 물리 GPU
시간 공정성은 아직 인수하지 않았다. Windows/WSL·CPU 자원·batch/runtime도 다르므로
결과는 로컬 배치 구성 벤치마크이며 정식 강도·순수 탐색 알고리즘 우위가 아니다.

첫 native 설정 `max_batch=16,min_batch=1`은 FP32/FP16 모두 CUDA invalid argument로
실패했다. 실패 로그를 보존하고 **256/4**를 별도 설정으로 다시 잠가 성공했다.
두 값을 함께 바꿨으므로 특정 min_batch가 원인이라고 확정하지 않는다.

## 6. 개발 대국과 후속 작업

같은 BT4 FP32·장치에서 서로 색을 교환하는 개발 대국 4쌍/8판을 준비했다. 기존
개발 opening 목록 첫 4개·완전한 8-ply prefix, 수당 500 ms·별도 host/transport 여유
100 ms, 전체 1,200초·판당 총 256 ply·worker pair 1을 실행 전 manifest에 잠갔다.
현재 Available인 threefold/fifty-move를 자동 수락하며 불법 수·시간패·crash는 loss,
인프라 실패는 abort·원본 보존, cutoff는 Incomplete로 집계한다. 기존 개발 pool이며
독립 holdout이 아니다. 시계/자원의 위 공백 때문에 `formal_strength_eligible=false`다.
8판을 563.15초에 완료했고 RoveZero 관점 **1승·2무·5패, 득점률 25%**였다.
6판 checkmate·2판 현재 Available threefold 수락이며 시간패·불법 수·crash·cutoff 0이다.

| opening | RoveZero 백 / 흑 | 쌍별 점수 / 2 |
|---|---|---:|
| Pirc | 패 / 패 | 0 |
| King's Indian | 패 / 패 | 0 |
| Italian | 무 / 무 | 1 |
| French (fixture ID `rz-french-advance`) | 승 / 패 | 1 |

두 엔진 동시 resident의 전체 장치 관측 최대는 **3,136 MiB**다. RoveZero 520회 착수의
응답 중앙값/최대는 500.39/538.08ms, LC0 522회는 496.33/508.20ms다. 선언한
500+100ms 경계를 넘은 착수는 없었다. 모두 끝난 뒤 두 process exit 0·실제 종료,
RoveZero final drain confirmed와 VRAM 0 MiB를 확인했다. 전체 PGN **1,106 ply**를
독립 `python-chess 1.999/chess 1.11.2`로 시작 prefix·색·합법 수·최종 FEN·종료 결과
대조했다. A Rules의 기존 감사기도 정리한 4pair/8판을 모두 통과했다.

PGN 원본의 `{ comment }` 공백 때문에 strict A 감사기가 pair 2의 claim 종료 문구를
거부한 첫 실패를 보존했다. comment 양끝 공백만 제거한 정리본은 모든 header·착수·
결과가 원본과 같음을 독립 대조한 뒤 A로 다시 감사했다. parser 구현은 바꾸지 않았고
원본·정리본·첫 실패·재검사 receipt를 함께 보존한다. A의 reason 공백 처리 개선은
별도 E 후속 항목이다.

RoveZero 대국 process 집계는 D Computed 16,160·B guarded root 520/non-root 15,640,
final drain-discarded 120·delivery drop 0·fatal/boundary/poison 실패 없음이다. stderr의
Expired/Stale는 그대로 남겨 두며 집계 성공을 모든 root의 물리 시계 인수로 확대하지
않는다. 4개 개발 opening·cross-OS·host 여유를 포함한 이 결과로 Elo/승격을 판정하거나
모델 자체의 강도를 분리했다고 하지 않는다.

| 담당 / 다음 순서 | 구현·인수 조건 |
|---|---|
| B·총괄 | 위 mate-in-one을 실제 root stats·확정 terminal 발견 여부로 재현한다. 확정 승패의 root 선택/solver 변경은 S 변경으로 별도 revision·CONTROL-0→1·전술 회귀·같은 weights 대국으로 검증한다. |
| B·D·총괄 | 독립 마감/stop에서 bestmove와 물리 GPU 완료의 순서, 현재 game/root별 accepted visits·fresh/cached 실행과 지연을 계측한다. 시간 여유 확대만으로 원인을 숨기지 않는다. |
| C·D | BT4 B1/FP32의 준비·대기·전송·GPU·backup 비용을 분리한다. batch/FP16/Graph 변경은 한 번에 하나씩 별도 수치·수명·VRAM 인수를 거친다. |
| E·총괄 | 기존 E CUDA launch V1은 arena/session 1 GiB·ONNX artifact 16 MiB로 닫혀 있어 이번 BT4 manifest/741 MB export를 인수하지 않는다. 별도 BT4 실행 profile/schema·artifact bounds·수명/receipt 연결을 수동 대조한 뒤 같은 OS·CPU 자원·정밀도·whole-wall 시계·GPU 물리 fence·holdout·사전 통계를 잠근다. 이번 개발 pair는 직접 native CLI의 별도 runner이며 E 정식 launch 통과가 아니다. |
| F·C | 권리와 실제 trainer/export round-trip을 확인한 뒤 학습을 검토한다. 모델 변경과 탐색 개선의 효과를 한 실험에 섞지 않는다. |

원시·정리 근거의 논리 루트는 저장소 밖
`reports/coordinator-integration/bt4-benchmark-20261004/`다. 고정 모델/manifest,
참조 fixture·오차, failed native 설정, 54개 query/log/VRAM 표본, binary/source pin,
startup/termination·감사·PGN을 보존한다. 코드·가중치·raw PGN을 문서에 내장하지 않는다.

## 7. 학습 계획과 완료 경계

동결 BT4의 로컬 실행은 확인했지만 **BT4 개별 학습·수정·재배포 권리는 미확인**이다.
현재 F의 `LinearFixture`는 실제 BT4/Maia/T1 trainer가 아니다. 6 GB 추론 성공을 BT4
학습 적합성으로 사용하지 않는다. BT4 학습을 실행하기 전에 원본 tensor의 trainable
복원·frozen round-trip·gradient/optimizer·checkpoint/resume·ONNX parity·optimizer
메모리와 microbatch를 따로 인수한다. 권리가 확인된 T1은 작은 대안으로 보존하며,
기존 구조 미세조정과 작은 학생 증류는 독립 후속 경로로 유지한다.

같은 weights의 탐색 CONTROL-0→1을 먼저 비교하고 같은 S1에서 학습 CONTROL-1→2를
분리한다. 교사 raw policy·search visits·WDL·cp의 의미, game/opening split·중복/전이
누출·validation/holdout과 checkpoint 선택을 잠근다. 256예제/200 step/20분은 과거
학습 smoke 제안이며 이번 실행값이 아니다. 실제 학습·교사 dataset 생성·새 유료 GPU는
실행하지 않았다. 세부 조건은 [TRAINING-PLAN](../TRAINING-PLAN.md)을 따른다.

로컬 전체 workspace all-target/all-feature test와 strict Clippy, release native build는
통과했다. 소스 `78b7c53`의 [두 OS CPU CI](https://github.com/daejunnom/RoveZero/actions/runs/37208852788)도
SUCCESS를 직접 조회했다. CI는 로컬 실제 CUDA·개발 대국의 대체 증거가 아니다.
C/A/D 수치 gate와 이후 UCI/예산/완료 순서의 소스 pin을 구분하며 앞 gate의 재사용은
영향 경로·feature·의존의 동일성 확인에 한정한다. 상세 상태는
[INTEGRATION-STATUS](../INTEGRATION-STATUS.md), 수동 연결은
[CONTRACT-ADOPTION](../CONTRACT-ADOPTION.md)을 따른다.
