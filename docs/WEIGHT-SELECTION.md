# 첫 가중치 호환 대상 조사와 구현 계약

기준일: 2026-10-02. 사용자 요청은 권리와 **RTX 4050 Laptop GPU 6 GB** 호환성을
조사한 뒤 첫 가중치 한 종류를 선정하는 것이다. 이 문서는 담당 C의 구현 C02/C03에
적용한다. 핸드오프 후보 `C02`의 recurrent 모델 연구와 구현 작업 `C02`는 서로 다른 ID다.
공통 계약은 작업 총괄이 소유하며, 모델별 표현을 공통 탐색 타입에 고정하지 않는다.

## 현재 선정 상태

첫 호환 구현의 **단일 가중치를 CSSLab Maia1의 공식 v1.0 `maia-1900.pb.gz`로 선정한다.**
6개 SE residual block, 64 channel, classical 112-plane 입력과 policy/WDL head를 갖춘
작은 LC0 호환 네트워크다. **인간의 수를 예측하는 모델이며 호환·런타임 기준선에 사용한다.**
프로젝트의 LC0 대국 강도 목표를 위한 최종 최강도 가중치 선정과 구분한다.
[공식 모델 설명](https://github.com/CSSLab/maia-chess),
[공식 v1.0 release](https://github.com/CSSLab/maia-chess/releases/tag/v1.0)

원저자 저장소의 MEMBER `reidmcy`는 weights와 code가 모두 GPL이라고 명시했다. 공식
release의 LICENSE는 GPL v3이고, 2026년의 후속 답변도 GPL 3 release라고 설명한다.
코드 라이선스를 가중치에 자동 적용한 추정과 구별되는 직접 근거다. 따라서 **GPL 조건을
따르는 로컬 사용·변환·수정의 첫 호환 대상**으로 채택한다. 재배포는 아래 의무를 충족한
산출물만 대상으로 하며, 모델이나 변환본을 MIT라고 표시하지 않는다.
[weights 적용을 명시한 원저자 답변](https://github.com/CSSLab/maia-chess/issues/8#issuecomment-744732973),
[GPL 3 release 후속 답변](https://github.com/CSSLab/maia-chess/issues/76#issuecomment-4982341747),
[v1.0의 LICENSE](https://github.com/CSSLab/maia-chess/blob/37de81e2bef89336e03266b3b5f7e1155ba68f5d/LICENSE)

구조와 입력이 작고 명확하여 자체 Rust 탐색과 모델 추론 연결·수치 대조를 먼저 검증할 수
있다는 기술 판단이다. RTX 4050 6 GB의 CUDA 가능성과 파일 구조를 조사했으며, 실제 목표
GPU에서 추론·peak VRAM·속도 검증은 아직 수행하지 않았다. **선정 완료와 C03 실행 인수는
별도 상태**다. 먼저 조사한 T70 `703810`은 가중치별 권리 미확인으로 보류한다.

## 후보 비교와 범위

| 후보 | 공식 확인 정보 | 첫 구현에서의 판단 |
|---|---|---|
| Maia1 v1.0 `maia-1900.pb.gz` | 6 block × 64 channel; 직접 검사한 classical/SE/policy/WDL; 원저자의 weights GPL 적용 명시 | **단일 선정**. 호환·런타임 기준선용 인간 수 예측 모델. 실제 GPU 실행과 대국 강도는 미검증 |
| T70 `703810` | 10 block × 128 channel; 단일 파일의 SE/입력/head 메타데이터 확인 | 권리 미확인으로 보류. 내장 license가 없고 공식 배포 정보에서도 가중치 적용 허가를 확인하지 못함 |
| T1-256x10-distilled-swa-2432500 | 공식 표의 GPU 메모리 1.6 GB, 파일 30~40 MB | 후속 후보. 공식 표의 메모리 값을 자체 Rust backend의 보장치로 쓰지 않음. 이 파일의 구조·권리는 이번에 개별 확인하지 않음 |
| BT4-it332 | 공식 표의 15 block × 1024 channel, GPU 메모리 4 GB, 파일 365 MB | 별도의 강한 외부 LC0 비교 후보. 첫 작은 호환 구현에서 attention 계열 전체를 함께 지원하지 않음 |

공식 메모리 표는 backend·batch·workspace·공동 점유 조건이 완전히 고정된 RoveZero 측정이
아니다. 특히 같은 6 GB GPU에서 두 엔진을 동시에 올릴 때의 합산 사용량을 보장하지 않는다.
[LC0 공식 Best Networks](https://lczero.org/play/networks/bestnets/)

## 고정할 원본 식별자와 실제 검사 범위

| 필드 | 값 |
|---|---|
| 공개 이름 | Maia1 `maia-1900.pb.gz`; targeted human rating 1900은 engine strength 보장치가 아님 |
| 원저자·공식 release | CSSLab; `v1.0` |
| release tag commit | `37de81e2bef89336e03266b3b5f7e1155ba68f5d` |
| 공식 원본 URL | `https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-1900.pb.gz` |
| gzip 파일 SHA-256 | `e2f565f42d7cd9f122557e6dc4eb84e5bbaedceda1d404dc485d3611c7c97a12` |
| 압축 해제 protobuf SHA-256 | `e8fe5a7f35594d4190c440a2716fda57ba9d071b26175d509476be7a4fda052a` |
| gzip 파일 크기 | `1,262,607 bytes` |
| 압축 해제 protobuf 크기 | `1,738,564 bytes` |
| 파일 내 license 필드 | **없음** — 외부 원저자 답변과 해당 release LICENSE를 근거로 사용 |
| 파일 magic / 최소 LC0 버전 | `0x1c0` / `0.21.0` |

먼저 T70 한 개를 조사한 뒤 권리 근거가 있는 작은 대안 Maia 한 개를 추가 조사했다.
각각 HEAD/공식 asset 정보로 크기를 확인하고, 파일별 50 MiB 상한 안에서 받아 gzip과
protobuf 메타데이터만 검사했다. tensor 수치를 출력·해석하거나 모델 로딩·추론·학습·
변환을 수행하지 않았다. 저장소 밖 `reports/weight-selection-2026-10-02/`에 원본 두 개와
`maia-1900-metadata.json`, `t70-703810-metadata.json`, `maia-license-evidence.json`을
보존한다. 기존 `models/`·`checkpoints/` 등 접근 제외 폴더는 조사하지 않았다.

gzip 재압축은 압축 파일 digest를 바꿀 수 있으므로 압축 파일과 protobuf의 digest를
서로 대체하지 않는다. 변환된 ONNX와 파인튜닝 파일에는 새 digest와 부모 원본 digest를
기록한다. T70 보류 기록의 protobuf digest는
`b30e742bcfd905815e0e7dbd4e1bafb41ade748f85d006b8e28758f1a3107ae3`, gzip digest는
`492e30de664d927cc155df20f1c18b0874addd18e921d9567c033fec9e6c4bfe`이며 크기는
각각 `7,580,184` / `6,433,429 bytes`다. 공개 filepath ID는 이 protobuf digest와 일치했다.
[T70 공식 legacy 목록](https://lczero.org/play/networks/legacy/),
[T70 개별 메타데이터](https://dev.lczero.org/networks/classic/t70/3810/)

## C02 모델별 형식 계약

아래 enum은 선정한 Maia 단일 파일에서 읽은 값이다. 이름과 의미는 고정한 공개 protobuf 명세와
대조했다. 명세 참조 SHA는 `c47d3683972d9ef293b0c0bc7675f7c2c5ce2274`다.
[net.proto 명세](https://github.com/LeelaChessZero/lczero-common/blob/c47d3683972d9ef293b0c0bc7675f7c2c5ce2274/proto/net.proto)

| 항목 | 후보 메타데이터 | 구현 지시 |
|---|---|---|
| 저장 encoding | `LINEAR16` = 1 | 저장된 16 bit 선형 양자화와 실행 dtype을 구별한다. 파일을 FP16 tensor라고 해석하지 않는다 |
| 입력 | `INPUT_CLASSICAL_112_PLANE` = 1 | `[batch,112,8,8]`; classical history 규약을 구현한다. canonical v2/Armageddon 입력을 묵시적으로 처리하지 않는다 |
| body | `NETWORK_SE_WITH_HEADFORMAT` = 4 | SE residual 6개, 64 channel, attention encoder 0개 |
| policy | `POLICY_CONVOLUTION` = 2 | convolution 결과를 LC0의 1858 수 표현으로 매핑한 뒤 합법 수 순서로 추출한다 |
| 출력/value | `OUTPUT_WDL` = 2 / `VALUE_WDL` = 2 | W/D/L 세 값. 요청의 side-to-move 관점과 출력 확률 의미를 검증한다 |
| moves left | 필드·head 없음 | 해당 모델에 moves-left 출력을 요청하지 않는다. 실제 출력이 있는 모델은 별도 descriptor가 필요 |
| activation | 별도 필드 없음 | 명세 기본값인 ReLU를 적용하되 필드 부재와 명시 값은 manifest에 구분한다 |

미지원 enum·누락된 필수 구조·길이/shape 불일치·비유한 값·과대 할당은 로딩 경계에서
명시 오류로 거부한다. 다른 네트워크를 자동 검색하거나 정상 mock으로 대체하지 않는다.
모델 descriptor와 평가 manifest에 원본 digest, 변환 digest, 모든 input/head/precision
옵션을 넣는다. 공통 `EvalResult`는 **합법 수 순서의 policy 벡터**를 내보내며 1858 슬롯은
`rz-encoding`의 해당 모델 adapter 내부 계약이다.

### 입력과 이동 매핑의 확인 지점

LC0 v0.32.1의 공개 형식 동작을 독립 대조 기준으로 사용한다. 참조 commit은
`fd71a2d921b689c5f479d3227c3806c8e272d9c5`다. 아래는 호환해야 할 형식 사실이며,
GPL 구현 코드나 테이블을 MIT 소스에 복사·번역하라는 지시가 아니다.
[고정한 encoder 참조](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/neural/encoder.cc),
[board 좌표 정의](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/chess/board.h)

- 현재와 이전 최대 7개 상태의 13 plane씩 총 104 plane을 사용한다. 각 상태는 자기 6종,
  상대 6종 기물과 반복 표시를 담으며 모두 **현재 요청의 차례 관점**으로 맞춘다.
- auxiliary 104~107은 자기/상대 queenside·kingside 캐슬링 권리, 108은 현재 실제 색,
  109는 halfmove clock, 110은 0, 111은 1이다. classical의 clock을 임의로 `/100`하지 않는다.
- 흑 차례의 rank 반전과 색 관점을 함께 적용하고 file은 유지한다. classical에 별도의
  canonical 변환을 넣지 않는다. FEN 이전 이력은 알려진 이력으로 위장하지 않는다.
- history fill 방식과 앙파상 때문에 생성할 수 있는 제한된 이전 상태는 참조 설정과 함께
  명시한다. fill 방식 차이는 인코딩 ID와 cache key를 바꾼다.
- 이동 공간은 **1858**이다. 마지막 66 슬롯은 q/r/b 승격이며, knight 승격은 해당 일반
  from/to 슬롯을 사용한다. queen 승격을 기본 슬롯으로 가정하지 않는다. Rust 내부 수는
  네 종류 승격을 계속 구별하며 캐슬링·흑 승격·포획 승격도 왕복 대조한다.

위 지점은 합성 tensor fixture와 별도 참조 실행으로 대조한다. 전체 mapping table을
GPL 소스에서 붙여 넣지 않고 독립 생성 규칙을 문서화해 구현한다. 잘못된 수를 model
index만으로 합법화하지 않는다.

### raw 출력과 합법 정책

변환된 ONNX의 policy는 1858개 **logit**이고 WDL은 converter 기본 설정에서 softmax를
적용한 3개 확률이다. 따라서 WDL에 softmax를 중복 적용하거나 policy logit을 확률로
간주하지 않는다. 이름만 신뢰하지 말고 실제 export graph의 출력 의미를 검증한다.
[고정한 ONNX converter의 출력 정의](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/neural/onnx/converter.cc)

합법 수에 대응하는 logit만 요청 순서로 모아, 명시한 temperature로 안정적인 softmax를
적용한다. 첫 기준선은 temperature `1.0`을 사용하고 변경은 탐색 옵션 실험으로 기록한다.
LC0 기본 temperature와 같다고 가정하지 않는다. 합법 수 순서 변경 시 대응 값만 순열로
변해야 하며 value는 유지되어야 한다. terminal은 규칙 계층이 처리하고 빈 합법 정책을
정상 비종료 평가로 반환하지 않는다.

## 코드·가중치·데이터 권리의 독립 확인

| 대상 | 확인된 근거 | 남은 조건 |
|---|---|---|
| RoveZero 자체 코드 | 사용자의 독립 Rust 엔진·MIT 코드 방향 | 실제 도입 시 자체 LICENSE와 외부 의존 notice 작성; GPL 코드/테이블 번역 혼입 방지 |
| LC0 engine / converter | GPL-3.0-or-later; 일부 CUDA 연결 추가 허가가 명시됨 | 별도 참조/변환 도구로 버전·binary digest·해당 notice 보존; Rust 제품에 GPL 구현을 링크·복사하는 선택은 별도 검토 |
| `lczero-common` protobuf 명세 파일 | 파일 헤더에 GPL과 추가 허가가 있음 | 이 파일을 그대로 vendoring하거나 생성 소스에 포함할지는 총괄의 별도 의존·라이선스 검토 대상 |
| 선정 Maia1 `maia-1900.pb.gz` 원본 가중치 | 원저자는 weights와 code 모두 GPL이라고 명시; 공식 release LICENSE는 v3 | 외부 파일로 운용; 변환·수정본에도 해당 GPL 조건과 provenance를 보존. 재배포에는 아래 실제 source·notice 검토가 필요 |
| T70 `703810` 원본 가중치 | 공개 출처·digest·구조 확인; 내장 license 없음 | 권리자의 명시적 모델 적용 라이선스 또는 허가 근거 필요 |
| LC0 공식 학습 데이터 | 공식 글은 training collection에 ODbL을 명시 | 데이터 수집·파인튜닝 전에 실제 데이터별 권리·출처·귀속·파생 결과 조건을 별도 검토 |
| Maia 학습 데이터 | 공식 README는 Lichess 게임과 학습 재현 경로를 설명 | 기존 weights의 GPL이 새 학습 데이터 사용 조건을 대신하지 않음. F02는 사용할 실제 게임·opening·교사 자료의 권리를 별도 확인 |
| ONNX Runtime | 자체 LICENSE는 MIT | 채택한 실제 버전과 포함 third-party notices 검토 |
| NVIDIA CUDA / cuDNN | GPU backend에 필요한 별도 runtime | 실제 선택 버전의 설치·재배포 조건과 지원표 확인; LC0 예외를 RoveZero/NVIDIA 전체에 자동 적용하지 않음 |

[LC0 공식 라이선스](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/README.md#license),
[공식 training data 권리 설명](https://lczero.org/blog/2021/06/the-importance-of-open-data/),
[ONNX Runtime LICENSE](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)

Maia 가중치 권리 적용은 다음처럼 기록한다. GPL의 사용·변경 허가와 전달 시 의무를
구별하며, `rights-confirmed`는 허가 근거가 확인되었다는 뜻이다. `redistribution-ready`는
배포할 실제 산출물의 의무까지 충족한 뒤 별도로 설정한다.

| 행위 | 적용 조건 |
|---|---|
| 외부 원본 파일의 로컬 추론 | GPL 적용 출처·원본 digest·원저자 고지 보존. 자체 MIT Rust 코드에서 외부 자산을 로딩하는 설계이며 weights를 MIT로 재표시하지 않음 |
| 로컬 ONNX 변환 | 별도 GPL converter와 변환 명령·버전·출력 digest 보존. 변환본도 covered asset으로 관리하며 도구 LICENSE가 weights를 다시 허가한다고 해석하지 않음 |
| 로컬 파인튜닝 | GPL의 변경 허가 범위로 다룸. 학습 비용·예산·데이터 권리와 F02 실행 조건은 별도 충족; 이번 문서 작업에서 실제 학습을 시작하지 않음 |
| 원본·ONNX·파인튜닝본 전달 | GPL 사본·copyright/notice·변경 내용/날짜를 보존하고 covered asset을 GPL 조건으로 제공. 배포 형태에 필요한 preferred modification form·Corresponding Source와 제공 경로를 확정한 후 전달 |

GPL v3의 0·2·4~6장이 위 구분의 근거다. 비소스 형식 전달을 다룰 때 단순 LICENSE 링크를
source 제공의 대체물로 쓰지 않는다. 수정 가능한 checkpoint/표현, 해당 변환·학습 코드와
설정, 필요한 재현 자료 중 무엇을 Corresponding Source로 제공할지 실제 산출물별로
정하고, 데이터 자체의 공개 조건은 별도로 확인한다. 모든 학습 데이터가 자동으로 GPL이
되거나 무조건 재배포 대상이라고 단정하지 않는다.
[선정 release의 GPL v3 원문](https://github.com/CSSLab/maia-chess/blob/37de81e2bef89336e03266b3b5f7e1155ba68f5d/LICENSE)

RoveZero 자체 소스의 MIT 정책은 유지한다. GPL LC0 구현·매핑 테이블·생성 코드를 제품에
복사·번역·링크하지 않고, 참조/변환은 별도 실행 파일의 파일 입출력으로 격리한다. 원본과
파생 가중치는 저장소 밖 asset로 보존한다. 이 경계가 결합 배포의 법적 판단을 자동으로
끝내는 것은 아니다. 제품+weights 번들 또는 GPL backend 링크를 실제로 선택하면 독립
저작물의 aggregate인지 파생·결합 저작물인지와 제공 의무를 총괄이 그 구성으로 검토한다.
GPL은 독립 저작물 aggregate와 covered derivative를 구별한다. 단순 파일 분리만으로
전체 배포의 MIT 보장을 주장하지 않는다.

권리 확인 기록에는 정확한 모델/digest, 권리자와 출처, 원문 라이선스 URL 또는 허가
근거, 로컬 추론·ONNX 변환·파인튜닝·원본/변환본/수정본 재배포 각각의 판단을 넣는다.
다운로드 가능·공개 GitHub·엔진 라이선스·변환 성공을 이 근거의 대체물로 쓰지 않는다.
권리자에게 질문을 보내는 행위는 사용자에게 명시적으로 요청받은 뒤에 수행한다.

보류한 T70를 후속 강도 가중치로 검토할 때 필요한 질의는 다음과 같다: **“T70 최종
`703810`의 위 protobuf digest를 자체 작성
Rust 엔진에서 추론하고 ONNX로 변환하며 파인튜닝할 수 있는 라이선스는 무엇인가?
원본·ONNX 변환본·파인튜닝 가중치 재배포 조건은 무엇인가?”** 실제 답변 또는 권리자의
공개 근거를 받기 전에는 `rights-confirmed`로 표시하지 않는다.

## RTX 4050 6 GB의 기술적 가능성과 미검증 범위

A~F의 개발은 GPU가 할당되지 않을 수 있는 클라우드에서 진행한다. RTX 4050 6GB는
별도의 목표 검증 환경이며 모든 개발 세션에 해당 장비가 있다고 가정하지 않는다.
담당 C는 GPU 없이도 loader·encoding·backend 코드·수치 fixture·검사 진입점과
가능한 CPU 신경망 참조를 개발할 수 있다. CUDA capability 부재와 GPU 검사의
skip·미실행은 명시하며 CPU/mock 통과를 목표 GPU 통과로 기록하지 않는다.

NVIDIA는 RTX 4050을 compute capability 8.9에 포함하고 Laptop GPU의 memory size를
6 GB로 명시한다. 이는 CUDA 계열 backend를 검토할 하드웨어 근거다.
[NVIDIA CUDA GPU 표](https://developer.nvidia.com/cuda/gpus),
[NVIDIA RTX 40 Laptop 사양](https://www.nvidia.com/en-us/geforce/laptops/40-series/)

선정한 Maia의 6×64 SE 구조와 작은 저장 파일은 첫 추론 대조에 적합하다고 추론한다. LINEAR16
payload를 FP32로 풀었을 때 가중치 tensor만의 크기는 protobuf 전체 바이트의 두 배보다
작다는 단순 상한을 계산할 수 있지만, activation·cuDNN workspace·allocator·batch·두
엔진 공동 점유를 포함하지 않는다. **파일 크기만으로 6 GB 실행 성공을 확정하지 않는다.**

실제 CPU 경로는 Rust `ort=2.0.0-rc.10`·ONNX Runtime 1.22.0·명시 CPU EP로 고정해
참조 대조와 root UCI 연결을 인수했다. 후속 첫 목표 GPU 검사는 명시 CUDA EP와
ORT 1.22.0·CUDA 12.8 계열·cuDNN 9.8.0.87·driver 610.62의 파일 identity를 잠가 실행했다.
수치·provider probe의 실제 근거와 남은 A/D GPU 연결은 아래 후속 인수에 구분한다. driver가
지원하는 CUDA 최대값을 설치된 Toolkit/cuDNN 존재 증거로 쓰지 않는다.

현재 ORT 문서는 release별 CUDA/cuDNN 호환표를 제공하며, cuDNN major version 간
호환을 보장하지 않는다. 따라서 구현 때 최신값을 추정하지 말고 선택 release의 표와
실제 DLL/공유 라이브러리 로딩을 확인한다. `gpu_mem_limit`은 CUDA EP arena의 상한이며
프로세스 또는 전체 GPU 메모리 상한이 아니다. TF32 기본 사용 여부도 precision manifest에
명시해야 한다. [ORT CUDA EP 요구·옵션](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

GPU를 요구하는 모드에서 CUDA EP가 없거나 초기화가 실패하면 `BackendUnavailable`로
종료한다. GPU 성공으로 기록하면서 CPU로 조용히 처리하지 않는다. 명시 CPU 모드는
다른 backend/run ID로 수행한다. 모델 가중치의 F32/F16/BF16 저장·실행 정밀도, TF32,
graph optimization 및 convolution 설정은 각각 기록한다.

## C02/C03 착수·인수 순서

1. 총괄이 공통 model/encoding descriptor와 rights evidence 형식을 정한다. 담당 C는
   선정한 Maia의 compatibility adapter와 합성/독립 참조 검사 계획을 작성한다.
2. 위 원저자 답변·release LICENSE·digest를 manifest에 잠가 Maia를 `rights-confirmed`로
   기록한다. 원본·변환본·수정본의 `redistribution-ready`는 실제 고지·source 제공 경로를
   검토한 뒤 따로 기록한다. 다른 가중치에 Maia의 허가를 적용하지 않는다.
3. 별도 GPL 참조 도구로 사용할 경우 위 LC0 v0.32.1 commit과 실제 실행 binary SHA를
   기록한다. `leela2onnx`를 외부 변환 도구로 실행할 수 있지만 출력 가중치가 MIT로
   바뀌었다고 표시하지 않는다. RoveZero 제품은 LC0 UCI를 신경망 API로 숨겨 호출하지 않는다.
4. 먼저 원본 LINEAR16의 해석을 보존한 FP32 ONNX, opset 17, 동적 batch의 입력/출력
   이름·shape·dtype·의미를 고정한다. converter 명령/옵션과 output digest를 기록한다.
   변환은 학습이 아니며 원본과 변환본 수치 대조 없이는 호환 완료가 아니다.
   [변환 옵션 참조](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/tools/leela2onnx.cc)
5. synthetic fixture와 참조 입력을 대조한 뒤 CPU 참조 → CUDA batch 1 → batch 2/4/8/16
   순서로 검사한다. 첫 프로파일은 inference worker 1개로 제한한다. precision별 허용
   오차·fixture 수·warmup/측정 횟수·최대 시간·메모리·출력 상한을 실행 전에 잠근다.
   첫 FP32 대조에서는 TF32를 끄고, FP16/TF32는 별도 수치·속도 실험으로 다룬다.
6. 백엔드 등록뿐 아니라 실제 GPU 실행/완료를 확인하고, CPU 준비·전송·GPU·수신·policy
   추출을 포함한 전체 지연과 P50/P95/P99·peak VRAM·batch 분포를 기록한다. 명시 batch
   상한에 맞지 않으면 해당 설정의 실패로 남긴다. OOM 뒤 자동 batch 축소나 무한 재시도를
   성공으로 기록하지 않는다.
7. 현재/흑 차례, 짧은/충분한 이력, 반복, 동일 보드·다른 이력, 앙파상, 양쪽 캐슬링,
   네 종류 포획·비포획 승격, policy 순열, batch 독립성을 검사한다. raw logit과 WDL,
   합법 수별 정규화 policy를 각각 비교하고 NaN·shape·미지원 모델 실패를 검사한다.
8. C03 인수는 **선정한 한 종류·고정한 backend/precision에서의 수치 호환과 실제 GPU
   실행**까지다. 모든 LC0 모델 지원·대국 강도 향상·학습 성공은 별도 증거를 요구한다.

위 CPU 참조 단계까지는 가능한 클라우드 환경에서 먼저 완료하고 코드·수치 검사와
GPU 인수 상태를 나눠 공유한다. CUDA 단계는 지원 환경에서 총괄 I02가 실제 실행·
메모리·수명·계측 근거를 별도로 인수한다. GPU가 없는 개발 환경에서 C03 전체를
불능으로 취급하거나 GPU 인수를 완료로 표시하지 않는다. 이 문서 공유는 새 실제
실험·유료 GPU 사용을 승인하지 않는다.

초기 FP32 수치 인수안은 raw policy logit에 `atol=1e-4, rtol=1e-3`, WDL과 합법 수별
확률의 최대 절대 오차에 `1e-4`를 사용한다. 이 기준은 아래 고정 CPU 참조에서 실제
검사했으며 GPU·다른 모델의 성능 보장치가 아니다. 담당 C와 총괄은 참조·fixture·precision을 함께 고정한다.
실패하면 원인을 먼저 구분하고, 오차 기준 변경은 근거·영향·이전 실패를 남겨 결정한다.

## 대국·파인튜닝에서 효과를 분리하는 기준

RoveZero 내부 비교는 원본 고정 가중치+기준 탐색 → 같은 가중치+새 탐색 → 같은 새
탐색+파인튜닝 가중치 순서다. 내부 A/B에서 backend·precision·resource·시간을 바꿔
탐색 효과에 섞지 않는다. LC0 대국은 외부 엔진 성능 비교이며 별도 manifest를 가진다.

같은 Maia를 LC0 참조와 RoveZero에 넣어 신경망 연결을 대조한 결과는 **호환성·인간 수
예측 모델의 작은 기준선 비교**다. LC0의 높은 강도를 겨냥한 훈련 모델이 아니므로 여기서
얻은 탐색·runtime 효과를 최신 강한 LC0 대비 목표 달성으로 해석하지 않는다. 강도 판정용 LC0
모델은 E01에서 별도로 고정하고 실제 6 GB 자원·동시 점유·시계 조건을 검증한다.

F02는 원본 가중치 권리, 학습 데이터의 별도 권리, 선택 구조의 checkpoint 복원,
policy index·WDL 관점·누출·seed 계약이 충족된 후 진행한다. F03의 관계 입력·latent·
반복 구조가 고전 SE 가중치와 호환되지 않으면 부분 초기화·증류·별도 학습 과제로
분리한다. 기존 가중치를 그대로 넣을 수 있다고 가정하지 않는다.

## 현재 완료 근거와 미완료 조건

- 완료: 공식 후보·사양·형식·라이선스 조사, 두 개 작은 공개 파일의 제한된 metadata
  검사와 digest 검증, 원저자의 weights GPL 적용 근거 확인, **Maia1 v1.0 maia-1900 단일
  호환 선정**, C02/C03 구현·인수 경계 작성.
- 후속 실제 CPU 인수: 고정 LC0 `fd71a2d921b689c5f479d3227c3806c8e272d9c5` converter·
  Eigen 원본 protobuf 참조, ONNX Runtime 1.22.0 CPU/FP32와 Rust loader/backend를
  `a9a5ae7f5c0bb5b13d8082e2ce662f8e807d8248`에서 연결했다. 직접 tensor 대조와 batch
  1/2/4/8/16, 실제 A 이력·합법 수 12개를 통한 C→D 수치 대조·drain, 실제 CPU UCI를
  검사했다. source·export·runtime·reference digest와 오차·범위는
  [통합 인수 상태](INTEGRATION-STATUS.md)에 기록한다.
- 후속 목표 GPU 부분 인수: `bde687c7f269af3c7cd501201edde14c5c8ef642`에서 RTX 4050
  Laptop GPU의 CUDA/FP32·TF32 off·CPU fallback 금지로 12개 원본 참조와 batch
  1/2/4/8/16을 대조했다. 실제 warm profile은 CUDA 98·CPU 0 kernel 이벤트이며 C worker도
  통과했다. 같은 최종 binary의 CPU 회귀도 통과했다. 전체 19-file native bundle·오차·
  장비·자원·실패·profile·source SHA는 [통합 인수 상태](INTEGRATION-STATUS.md)에 둔다.
- 미완료: 실제 A/D GPU 요청·수명, process peak VRAM·GPU 종단 성능, 정식 대국·실제 Maia 파인튜닝,
  외부 배포 형태의 source/notice 충족 확인. T70 권리는 보류 유지.
- 다음 작업: 총괄 I02가 실제 A/D GPU 연결·D02 종단 계측과 E 강도 gate를 각각 인수한다.
  학습·장시간 대국의 구체적 예산은 실행 전에 확정한다.
