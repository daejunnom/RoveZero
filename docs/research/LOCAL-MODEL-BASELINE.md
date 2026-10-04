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
| B·총괄 | 8장의 mate/terminal 회귀와 직접 자식만 대상으로 하는 S1 인수를 참조한다. 전체 solved propagation·mate-distance와 같은 weights 동일 시간 대국은 별도 후속이다. |
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

## 8. 종료 회귀·조건 대조·B1 계측과 별도 S1

2026-10-05 사용자 지정 순서로 종료 회귀→입력·평가 조건→동일 바이너리 A/A와
B1 계측을 진행했다. 이어 사용자가 메이트 오인 위험과 LC0 정책 확인을 지정하여
정확한 종료 사실만 소비하는 final-selection을 별도 **S 변경**으로 연결했다.
이 장의 S0/S1은 **최종 착수 정책**이며, 앞선 Maia 폭 1/4 실험의 S0/S1이나
OPT-12 다중 요청 선택 정책과 혼동하지 않는다. 여기서는 양쪽 모두 동일한 PUCT,
batch 1, 동결 BT4, FP32·TF32 off, raw cache off다.

### 8.1. LC0의 증명과 추정치, RoveZero의 구현 범위

고정 LC0 `fd71a2d921b689c5f479d3227c3806c8e272d9c5` classic search를 확인했다.
실제 legal moves가 없고 check인 상태를 terminal win으로 만들며, 최종 후보 정렬에서
방문된 terminal win/loss를 일반 신경망 Q와 구분한다. LC0의 edge 관점과 RoveZero의
child side-to-move 관점을 혼동하지 않는다. 승리끼리는 짧은 mate distance,
패배끼리는 긴 거리를 고른다. bounds 전파는 자식 하나의 높은 Q와 다르며, 승리는
확정 winning child로, 패배는 모든 legal child의 확정 bound로 증명한다.
[종료 생성](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc#L1920-L1938),
[최종 후보 정렬](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc#L758-L821),
[bounds 전파](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc#L2285-L2350)

RoveZero는 GPL 소스나 전체 LC0 solver를 복사하지 않고 최소 범위를 독립 구현했다.
`--final-selection=visits`는 기존 S0이며 기본값도 그대로다.
`--final-selection=exact-terminal`은 `exact-terminal-child-v1` S1이다. 실제 Rules
분류와 마지막 consume guard를 통과해 **직접 자식 Node::Terminal로 승인되고 방문된**
edge만 정확한 승패로 정렬한다. 자식의 side-to-move 값 -1은 root 승리이며 +1은
root 패배다. 나머지는 기존 방문 수→Q→legal 순서로 정렬한다. Q=1/-1, 미방문 자식,
거부·취소된 완료, 일부 winning descendant만으로 solved parent를 만들지 않는다.
정책은 첫 selection 전 고정하며 선택 결과의 policy identity에 별도 suffix를 남긴다.

이번 변경은 직접 자식의 확정 종료를 최종 선택에 반영한다. 전체 subtree의 승패
bounds 전파·메이트 거리·selection 조기 중단은 구현하지 않았다. 따라서 높은 Q를
메이트로 표시하거나 긴 강제 메이트를 모두 증명하는 기능으로 설명하지 않는다.
종료 값은 정확한 Rules 계약의 결과이며 신경망 WDL과 별도로 보존한다.

4장의 고정 artifact/runtime 인자 뒤에 `--final-selection=visits` 또는
`--final-selection=exact-terminal`을 명시하여 실행한다. 진단 example은 동일한
native 인자를 사용하며 `--probe-cases=terminal --probe-wall-ms=30000
--search-simulations=4096`으로 종료를, `--probe-cases=profile --profile
--probe-wall-ms=30000 --search-simulations=128`으로 두 opening의 B1 host journal을
실행한다. `--probe-cases=parity --search-simulations=1`은 실제 네 opening의
root tensor/평가 대조용이다. 각 실행은 새 외부 output root를 사용한다.

### 8.2. 실제 BT4 종료 회귀와 같은 바이너리 S0/S1

새 진단 소스 `cf94d07f26cbc1ce5de72f94003938512bad7290`에서 legal 후보·실제 자식
classification·승인된 terminal backup의 경로와 root visit/value delta를 관측했다.
관측은 승인 후의 단일 slot이며 search 권한이나 새 방문을 만들지 않는다. 신경망을
주입한 인공 트리에서는 Q=1과 winning descendant를 오인하지 않는 경우, 마지막
guard가 거부한 terminal, 정책 변경 거부도 검사했다.

| 고정 사례 | S0 최종 착수 | S1 최종 착수 | 실제 판정 |
|---|---|---|---|
| 백 mate-in-one | `g6f6` | `g6h5` | S0는 ongoing, S1은 실제 checkmate |
| 흑 mate-in-one | `g3f3` | `g3h4` | S0는 ongoing, S1은 실제 checkmate |
| 백/흑 stalemate root | 착수 없음 | 착수 없음 | draw, 평가 제출 0 |
| 백/흑 이미 checkmate root | 착수 없음 | 착수 없음 | 현재 차례 패배, 평가 제출 0 |
| 백/흑 패배 회피 | `g8f8` / `g1f1` | 동일 | queen capture 뒤 dead-position draw |

메이트 정답은 양쪽 각각 네 개다. 특정 UCI 한 수 일치를 요구하지 않고 실제 적용한
자식의 checkmate 여부로 판정했다. 패배 회피 fixture의 다른 수는 상대 mate-in-one을
허용한다. CPU/mock의 두 정책×8개 검사와 실제 C/D CUDA 두 정책×8개를 분리해 기록했다.

S0에서도 네 즉시 메이트가 legal 후보에 있고 checkmate로 분류됐으며, leaf -1이
root +1로 한 번씩 backup됐다. 백 `g6f6`은 451회 방문·Q≈0.999985, 직접 mate
`g6h5`는 246회·Q=1이라 방문 우선 최종 선택에서 밀렸다. 따라서 이 표본에서
Rules·관점·중복 backup 계약 위반은 발견하지 않았으며 final-selection 변경은 S다.

S0/S1 비교는 소스 `9817647465cdd6d638f04023b2e49139b35ed57e`, 동일 binary SHA-256
`581d41621de3e841add3575a8440c393e1448e968f08976615c9f46673cabd8d`,
explicit **4096 simulations·30초 case cap**에서 정책 옵션 하나만 바꿨다.
두 실행의 입력 tensor·legal indices·root policy/WDL·전체 root statistics·카운터와
terminal 경로가 정확히 일치했다. 각 정책의 **16,005 terminal backup**에 대해 root
visit delta=1·경로 길이에 따른 부호를 확인했고 오류는 0이었다. 독립 `chess 1.11.2`
감사로 모든 후보와 종료 경로를 다시 재생했다. 매 case의 4096 완료 방문을 새 NN
실행 수로 부르지 않는다. 양쪽 actual process는 383회 NN 완료·정상 종료를 기록했다.

S0 재시도의 첫 실행은 로컬 host 디스크 부족으로 runtime bundle 동기화 단계에서
실패했으며 비교 표본에서 제외했다. 실패·부분 복사 목록·exit 1은 보존했다. pinned
원본 라이브러리와 일치하는 비활성 복사본만 hash 검증 후 정리하고 새 namespace에서
동일 설정으로 재실행했다. weights·입력·batch를 바꾸거나 실패를 승리로 집계하지 않았다.
단일 표본의 경과 시간 차이를 S1 속도 개선으로 해석하지 않는다.

일반 `rz-uci`의 실제 threaded worker/CLI 경로도 수정 소스 `fabe88e86f27360a8a6172da5ce4573f7592d481`,
binary `2bcb3d5db53ac2edcd45c562202fc031b58173b8855cf9966a963a19d03cffec`에서
`--final-selection=exact-terminal --search-simulations=4096` 및 `go nodes 4096`으로
같은 여덟 상태를 실행했다. 백/흑 즉시 메이트와 두 회피 수가 위 표와 일치하고
종료 root 네 개는 `bestmove 0000`을 반환했다. 독립 oracle 감사·native startup/termination·
process exit 0·physical drain을 확인했다. example의 관측 경로와 일반 UCI 결과를 구분한다.

### 8.3. 이력·입력·legal policy·WDL 대조

BT4 원본/ONNX/manifest digest는 4장과 같다. 이전 실제 대국의 LC0 `HistoryFill=no`
및 RoveZero No, explicit simulation cap 4096을 manifest와 실행 argv로 재확인했다.
짧은 네 terminal fixture·startpos·Ruy 5-ply·이전 실제 대국 네 opening의 완전한
8-ply prefix, **10개 표본**을 actual Rust root 평가와 고정 LC0 Eigen FP32로 대조했다.
모든 112×64 tensor float bit와 ordered legal policy index가 정확히 일치했다.
temperature=1.0에서 policy 최대 차이는 **3.831388086e-6**, WDL은 **4.619359970e-7**로
사전 `1e-4` 오차 안이다. 이는 표본 대조이며 이전 대국 모든 root를 다시 평가한 것은 아니다.

LC0 Python binding의 `GameState.as_input`은 FEN_ONLY를 고정한다. 짧은 No 참조는
독립 LC0 tensor를 생성한 뒤 pinned encoder 규칙대로 없는 13-plane history block만
0으로 바꿨다. 해당 네 fixture에는 EP가 없고 missing history 범위를 확인했다.
No 옵션을 직접 제공하는 binding으로 검사했다고 표현하지 않는다. 실제 8-ply opening은
8개 known frame이 있어 padding 변환이 없다. 앞선 No 5/Repeat 7 수치 gate는 영향
encoding/evaluator source의 동일성을 확인한 범위로 재사용한다.
[binding 입력](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/python/weights.h),
[짧은 이력 encoder](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/neural/encoder.cc#L235-L300)

별도 조건 불일치는 **policy temperature**다. 이전 LC0 대국은 기본 **1.36**,
RoveZero는 **1.0**이었다. 원본 로그와 설정은 보존하고 공통 temperature에서의 수치
대조와 이전 실제 priors의 차이를 분리했다. 네 opening에서 실제 온도 차이에 따른
합법 policy 최대 차이는 0.04253~0.18797이다. HistoryFill 오류나 기본 128을 원인으로
지목하지 않으며, 이전 8판을 입력·탐색 조건이 모두 일치한 대국으로 재표기하지 않는다.
[LC0 옵션](https://lczero.org/play/configuration/flags/)

### 8.4. 먼저 동일 바이너리 A/A, 이후 B1 host 병목

이전 baseline source `78b7c53502cadaff77fc6de5f0832eee55b9938c`와 immutable binary
`1a4b976b4110d251e21b09ed1e1f29b398b4ed01f533788a941de8dc4be969b3`를 그대로 사용했다.
두 새 process × 세 반복 × 두 opening × nodes/movetime, 총 **24 query**다.
startup은 go 응답 시간에서 제외하고 CPU 2개·RAM 6 GiB·swap 0·단일 worker를 잠갔다.

| 조건·각 6표본 | 응답 평균 ms | 변동계수 |
|---|---:|---:|
| startpos, `go nodes 128` | 1,534.571 | 9.910% |
| Ruy-black, `go nodes 128` | 1,523.366 | 7.571% |
| startpos, `go movetime 1000` | 984.800 | 0.280% |
| Ruy-black, `go movetime 1000` | 985.653 | 0.335% |

여기서 128은 명시한 `go nodes` workload다. native configuration의 상한은 4096으로
잠갔으며 기본값 탓으로 해석하지 않는다. 고정 방문 작업에도 7.6~9.9% 편차가 관측되어
단일 측정의 작은 차이는 개선으로 채택하지 않는다.

최초 source-profile은 `cf94d07` 진단 example의 두 opening·각 128 non-root visits로 실행했다.
root 초기화 포함 실제 요청/완료/전달/소비가 **258회**, 모든 물리 batch가 1이다.
8192 한도 안의 8108 metadata record, complete timeline, producer join, 누락·중복·
identity mismatch·미소비·worker 실패 0을 확인했다. 계측은 실제 source clock을 쓴다.

그 뒤 일반 `rz-uci`의 threaded worker를 같은 두 opening·`go nodes 128`·configuration
cap 4096으로 다시 계측했다. source `fabe88e`, binary `2bcb3d5d…3cffec`의 실제
build features는 `onnx-cuda,experimental-batch`다. `experimental-notify`와
`experimental-best-move`는 꺼져 있어 **실제 기준 UCI도 1ms polling과 전체 outcome
진행 조회**를 사용한다. 코드에 완료 신호 기능이 존재한다는 사실을 이번 실행에서
사용했다는 증거로 바꾸지 않는다. 두 기능을 조용히 활성화하지 않았다.

일반 UCI의 8182/8192 metadata·complete timeline·join·실제 요청/물리 완료/전달/소비
258회·실패/미소비/누락/중복 0·physical drain을 확인했다. 다음 표는 이 **일반 UCI**
profile이며 처음 진단 example의 분포는 외부 원본에 따로 보존한다.

| 일반 UCI의 실제 host 단계 | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|
| Rules replay·legal 생성·history export | 0.042959 | 0.201463 | 0.305329 |
| C input encoding·hash·key 검사 | 0.030850 | 0.048060 | 0.073043 |
| worker의 실제 C 인코딩 | 0.024717 | 0.038974 | 0.056563 |
| runtime queue | 0.003220 | 0.006054 | 0.018149 |
| synchronous ORT Run 전체 | 10.089241 | 15.558466 | 23.129515 |
| 물리 완료→owner 관측 대기 | 0.601590 | 1.010235 | 1.082479 |
| root 초기화/guarded backup | 0.002605 | 0.003916 | 0.010656 |
| 최종 착수 조회 | 0.001549 | 0.003524 | 0.013862 |

최종 착수 조회는 pump 중 진행 조회를 포함한 3235개 표본이다. 나머지 표의 keyed
단계는 258개다. `StateReplay`에 Rules legal 생성·history export가 포함되며 별도
`LegalValidation`은 이미 만든 authority 확인이다. SearchPreparation/PhysicalWorker와
세부 단계는 중첩되어 분위수나 총합을 이중 합산하지 않는다.

두 일반 UCI의 go 응답 합계 3,022.374ms 중 inclusive ORT Run interval 합계는
2,813.345ms(**93.084%**), replay는 17.935ms(**0.593%**)였다. **이 두 opening/B1
표본에서 가장 큰 관측 host interval은
ORT Run**이다. 이것은 CUDA kernel만의 시간이 아니다. 전송·provider·동기화·호스트
작업을 포함하며, replay를 BT4의 최대 병목이라고 주장할 근거는 나오지 않았다.

일반 UCI의 전체 process 250ms GPU 표본은 94개·관측 최대 VRAM 1,127MiB·최대 사용률
78%다. 앞선 진단은 78개·최대 80%이며 모두 startup을 포함한다. 중앙 사용률을
search 사용률로 해석하지 않는다. cgroup의
RAM peak는 6 GiB·memory.max hit가 있었고 OOM/kill 0이었다. page cache·메모리 압박의
영향은 측정 조건으로 보존한다. 첫 profile 실행은 종료/drain 영수증이 회수되지 않아
미확정 실패로 남기고 fresh run의 성공으로 대체 기록하지 않았다.

GPU H2D/D2H와 kernel-only 시간은 **미측정**이다. 현재 고정 ORT bundle은 CUDA device
profiling 빌드가 아니고 nsys/ncu도 없다. 일반 ORT placement JSON의 `kernel_time`은
host node interval이며 실제 device trace로 승격하지 않는다. CUPTI와 CUDA profiling
빌드가 필요한 별도 진단 경로를 준비하되, 해당 경로를 기존 B1 production 측정으로
조용히 바꾸지 않는다. 후속은 transfer/kernel/wait를 같은 입력과 물리 completion
fence에서 대조하는 것이다.
[ORT profiling 요구](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)

### 8.5. 같은 입력·batch의 별도 추론 비교와 인수 경계

소스 `3ef917199453eda9fbd151aa9af85071e27efbe8`의 `maia_check --benchmark-rounds=20`과
pinned native LC0 `backendbench`를 별도로 실행했다. startpos/No의 동일 FP32 tensor
SHA-256은 `c33ae28cfa8081c3ff247f2ee36881e28520e4be8d585df726b8728b7dac22b6`다.
cache는 끄고 B1/B16 각각 20 round를 실행했다. native LC0는 threads 1,
min_batch 1/max_batch 16·step 15·batches 20·policy temperature 1을 명시했다.

| 실제 batch·20 round | RoveZero ORT host Run 평균 | LC0 CUDA backend 평균 | 완료 NN 항목 수/각 엔진 |
|---|---:|---:|---:|
| 1 | 11.087ms | 11.637ms | 20 |
| 16 | 76.947ms | 62.623ms | 320 |

이는 동일 입력·batch의 **배포 경로별 추론 측정**이다. Windows LC0와 WSL2 ORT,
runtime·CPU 한도·6 GiB RAM pressure 차이는 남아 있다. 기존 서로 다른 엔진의
nodes/NPS를 나눈 효율 지표가 아니며 같은 시간 대국의 기력 비교도 아니다. LC0가
backendbench에 표시한 throughput의 항목 수를 탐색 nodes로 바꾸지 않는다.
RoveZero의 기존 12-case 수치 및 B1/2/4/8/16 검사도 통과했다. 최초 LC0 호출은
지원하지 않는 `--config`를 거부했으면서 exit 0이었으므로 미실행으로 기록하고,
수정 호출의 두 expected batch 결과와 실제 정상 종료를 별도로 확인했다.

수정 소스의 전체 workspace all-target/all-feature **714 passed·0 failed**,
ignored 미실행, fmt/strict Clippy와 release build를 확인했다. Windows CI가 잡은
bounded Debug의 unused CUDA path는 `fabe88e86f27360a8a6172da5ce4573f7592d481`에서
private path를 출력하지 않는 bundle presence/digest로 보완했다. 실제 실행 소스와
문서 head를 구분한다. 해당 소스의 [CI 37218841079](https://github.com/daejunnom/RoveZero/actions/runs/37218841079)는
Windows·Ubuntu의 모든 필수 step SUCCESS를 직접 확인했다. 최신 CI 관측과 문서 변경의
검사 재사용 범위는 [통합 기록](../INTEGRATION-STATUS.md)에 둔다.

회수 논리 루트는 `reports/coordinator-integration/bt4-diagnostics-20261005/`다.
원본 query·terminal trace·입력·수치·AA/profile/inference·실패·supervisor·binary pin을
외부 보존한다. 이 인수로 전체 solver, 물리 deadline/stop race, E BT4 launch V1,
same-time holdout 기력·Elo·학습·RunPod 지원을 완료했다고 보고하지 않는다.


## 9. stop·마감 출력의 물리 완료 경계

2026-10-05 총괄은 저장 개선 뒤 남은 B03/C/D 수명 경계를 먼저 인수했다. 재개 시
원격 #20의 head `3ec439e3`·base `87017a27`·Draft/open·두 OS CI 성공과 리뷰 부재를
확인했다. 공통 계약 0.1·BT4 원본/ONNX/hash·FP32/TF32 off·HistoryFill No·B1·cache off,
기준 visit-first S0와 opt-in exact-terminal S1은 유지한다.

### 9.1. 재현과 변경

기존 자연 완료 경로는 runtime shutdown 뒤 완료를 전달했지만 독립 stop/deadline
소유자는 논리 취소 직후 착수를 출력했다. 물리 작업을 차단한 새 고정 검사는 이
응답 순서를 재현해 실패했다. 논리 취소만으로 물리 완료를 주장할 수 없으므로
`62719f5`에서 B의 production Owner에 출력 보류 경계를 추가했다.

- Session은 stop/hard deadline에서 기존 guard를 닫고 **마지막 유효 착수**를 고정한다.
  늦은 Progress/Complete로 그 착수를 교체하거나 새 backup을 승인하지 않는다.
- Owner는 소유한 worker 모두의 정상 drain·join을 확인한 뒤 한 번만 출력한다.
  기다리는 동안 입력·isready 처리는 계속하고 timer가 완료/시간 초과를 깨운다.
- 보류 출력은 game/root/model/encoding/backend scope에 묶는다. 받아들인 position/go,
  새 게임과 종료는 이전 보류 출력을 폐기한다. 중복 stop은 새 출력을 만들지 않는다.
- 같은 유한 `shutdown_limit`을 출력 대기에도 적용한다. drain 오류·panic·시간 초과는
  `PhysicalFenceFailure`로 전달하고 착수 승인을 내보내지 않는다. 원래 typed 원인을
  최종 serve 오류에 보존하며, 종료 중인 물리 worker와 pin을 임의 해제하지 않는다.

이는 B의 수명·출력 정확성 보완이다. PUCT selection/backup·방문 수·최종 S0/S1 정렬,
C input/WDL·D의 물리 lease·library cache를 바꾸지 않는다. E의 닫힌 V1 집계에
새로운 매 착수 완료 증명을 추가한 것으로 해석하지 않는다.

### 9.2. 실제 BT4 CUDA 인수

실제 source는 `3d90a0385245d3e77638186c97539c75c24bdd28`, binary SHA-256은
`f818756e37f8dca59915473be344ae331ad8d7bb57d91d17837cac12d85d737f`다.
features는 `onnx-cuda,experimental-batch`, notify/best-move 기능은 꺼진 기존
1ms polling이다. CPU 2·RAM 6 GiB·swap 0·pids 128, 전체 360초와 유한 query/quit
한도, 256 MiB 산출물 상한을 잠갔다. 실제 RTX 4050 6GB에서 세 프로세스를 차례로 실행했다.

| 실제 검사 | 결과 |
|---|---|
| S0 stop/deadline/root/newgame | 8 query, 물리 추론 53회·B 소비 48회(root 8/non-root 40)·완료 뒤 미소비 5회 |
| S1 같은 수명 검사 | 8 query, 물리 추론 54회·B 소비 50회(root 8/non-root 42)·완료 뒤 미소비 4회 |
| C/D와 실제 source journal | 각각 1829/1856 records·producer join·complete journal/accepted timeline, 누락/중복/identity mismatch/오버플로/물리 실행 중첩 0. 미소비 5/4와 drain discard 5/4가 일치 |
| 독립 착수 검사 | python-chess 1.11.2로 24개 응답의 합법/정확 종료 확인; 바뀐 root/game의 착수도 현재 상태에서 합법 |
| S1 종료 회귀 | 백/흑 실제 mate, 두 stalemate·두 checkmate root의 0000, 두 패배 회피의 실제 draw; 8개 go nodes 4096. NN 완료/소비 383·root 4/non-root 379 |
| 종료/실패 | 세 프로세스 exit 0·confirmed physical drain·runtime mapping/원래 service/collection 오류 없음, 남은 owned cgroup PID와 OOM/kill 0 |
| 저장 재사용 | 세 프로세스 모두 기존 CUDA 19-file runtime cache 재사용, 새 실행별 라이브러리 복사 0 |

두 수명 프로세스에서 실제 `go movetime 50`의 응답 네 개는 33.252~42.828ms다.
중단/교체 뒤 응답은 1.269~12.988ms다. 이는 표본의 전체 host 응답 관측이며
S0/S1 속도 우위나 모든 GPU/포지션의 마감 보장으로 일반화하지 않는다. source journal의
물리 실행 완료와 **실제 소비**를 구분해, 버린 9개를 새 방문/성공 소비에 더하지 않았다.
장치 전체 250ms 표본의 최대 VRAM은 1131 MiB이며 startup을 포함한다. 커널/전송의
별도 device 시간과 process peak는 여전히 미측정이다. 세 프로세스 종료 뒤 장치는
0 MiB·0%였다.

첫 측정 도구는 실제 native exit 0 뒤 영수증을 역할 root에서 찾았으나 실제 위치는
`native-process-<pid>`여서 도구 인수가 실패했다. 해당 supervisor/출력/실패를
`s0-fence`에 보존하고 위치를 맞춘 fresh 실행만 위 표에 인수했다. 처음의
movetime 5는 기본 출력·drain 여유보다 작아 유효 admission 구간이 없는 사례였다.
이를 진행 중 GPU 마감의 증거로 사용하지 않고, 변경한 명시적 50ms workload를 구분한다.

실제 source의 전체 workspace 검사는 733 passed·0 failed·16 ignored이며
fmt·strict Clippy·release가 통과했다. [CI 37237882649](https://github.com/daejunnom/RoveZero/actions/runs/37237882649)의
두 OS 필수 step 성공을 직접 확인했다. 후속 검사는 오류가 최종 serve Result까지
전달됨을 추가 확인하며, 테스트/문서 후속과 실제 GPU 실행 source를 구분한다.

원시 자료·명령·오류·binary/CI pin은 저장소 밖
`reports/coordinator-integration/physical-fence-20261005/`에 보존한다. 다음은 E의 BT4
artifact/bounds/profile과 같은 integration SHA의 C/D/E consumer를 맞춘 뒤, HistoryFill과
policy temperature 등 조건을 고정한 동일 시간 holdout pair다. 현재 검사를 E BT4 인수,
전체 solver·mate distance·device profiling·기력/Elo·학습의 완료로 승격하지 않는다.
