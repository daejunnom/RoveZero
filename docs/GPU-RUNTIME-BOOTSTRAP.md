# CUDA 런타임 부트스트랩과 인수 경계

이 문서는 총괄 I01/I02와 C03의 첫 Linux CUDA 검증에 적용한다. 공통 계약 revision
`0.1`은 유지한다. 기존 CPU 단일 ORT 파일 pin과 CPU backend identity를 보존하고,
CUDA 런타임의 추가 의존성과 오류 수명을 별도로 검증한다.

## 책임과 자산

`rz-eval`은 모델·입력·출력·provider·물리 실행을 소유하고, `rz-native-loader`는
검증된 네이티브 라이브러리의 로딩과 실제 프로세스 매핑만 담당한다. 새 crate를 두는
이유는 동적 로딩의 작은 `unsafe` 경계를 분리하면서 기존 평가 crate의
`forbid(unsafe_code)`와 공통 계약의 역의존 금지를 유지하기 위해서다. 로더는 체스,
탐색, 스케줄러, 모델 tensor를 알지 못한다. 총괄이 workspace와 직접 의존성을 소유한다.

첫 검증은 ORT `1.22.0`, Rust wrapper `2.0.0-rc.10`, CUDA 12.8 계열 라이브러리와
cuDNN `9.8.0.87`의 명시적 bundle을 사용한다. ORT core·shared provider·CUDA provider와
NVIDIA 의존 파일 각각의 이름·byte 수·SHA-256을 잠근다. 이 버전 선택은 모든 CUDA
장비·모델·정밀도의 지원을 의미하지 않는다. NVIDIA 파일과 해당 notice는 저장소 밖에
보존하며 로컬 검증 사용과 재배포 조건을 구분한다. Maia 원본·ONNX의 GPL 조건은
자체 MIT 소스와 분리한다.

## 로딩 전후 조건

- manifest는 미지원 schema·추가 필드·중복 이름·경로 성분·빠진 필수 파일·미지원
  basename·과대 할당을 거부한다. 첫 프로필의 NVIDIA 파일은 닫힌 집합으로 지정한다.
- 복사 manifest의 hash는 파일 identity이며 native 실행 허용 목록은 별도다. 로더는
  검토한 ORT 3개·NVIDIA 16개의 이름·크기·SHA-256을 소스의 고정 profile과 대조한다.
  같은 이름의 임의 ELF와 caller가 선언한 새 hash는 거부한다. 다른 release 지원은
  출처·권리·ABI와 실제 검사를 거친 소스 변경으로 추가한다.
- bootstrap은 새 private directory에 독립 inode로 streaming copy하고, 길이·hash를
  검증한 뒤 writer를 닫고 읽기 전용으로 바꾼다. 모든 OS file pin과 directory 소유권을
  유지한다. 기존 caller source를 이후 바꾸어도 로드할 사본은 달라지지 않아야 한다.
- 제한은 파일당 1 GiB, bundle 전체 4 GiB, 항목 32개다. 실제 자산 집합은 이 상한보다
  작아야 한다. 큰 네이티브 파일 전체를 메모리에 읽지 않는다.
- core hash만으로 CUDA 실행 identity를 만들지 않는다. 파일 역할·이름·길이·hash를
  명시적으로 직렬화한 전체 bundle digest를 CUDA backend identity와 process latch에
  포함한다. private 절대 경로를 공유 identity나 진단에 넣지 않는다.
- 로더는 이미 존재하는 외부 NVIDIA runtime 매핑과 모호한 로더 환경을 거부한다.
  검증한 절대 경로를 의존 순서대로 로딩하고 실제 매핑의 inode를 확인한다. ORT의 CUDA
  provider에 의존 검색 경로가 있다고 가정하지 않는다. 원본 ELF의 경로를 수정하지 않는다.
- host CUDA driver와 OS 표준 라이브러리는 별도의 플랫폼 근거다. 이를 다운로드한
  bundle 파일인 것처럼 보고하거나 runtime bundle의 hash로 대신 식별하지 않는다.
- 동적 로딩은 네이티브 코드를 실행한다. hash는 byte identity이며 sandbox나 신뢰할
  수 없는 네이티브 코드의 안전성 증명이 아니다. 부모 경로·프로세스·OS 사용자에 대한
  소유 계약을 전제로 하며 동일 UID의 악성 변경을 봉쇄한다고 주장하지 않는다.
- 네이티브 로딩을 시작한 후에는 부분 성공도 프로세스 수명 동안 보존한다. 실패를
  unload·재시도·다른 bundle 선택으로 자동 복구하지 않는다. 첫 실패와 원인을 latch하고
  새 실행을 거부한다. 정상 성공 후에도 probe가 추가로 로드한 vendor 파일을 재확인한다.

## CUDA 실행과 실패 수명

CPU fallback을 금지하고 FP32·TF32 off·명시 device·batch·thread·arena 조건을 사용한다.
provider 등록만으로 CUDA 실행을 인정하지 않는다. 실제 profile의 실행 kernel이 CUDA에
배치되었는지 확인하고, 같은 입력의 policy와 WDL을 독립 LC0 참조에 대조한다.

CUDA `Run` 오류나 panic은 GPU 동기화 완료의 증거가 아니다. 완료가 확인되지 않은
세션·입력·worker job·예약을 격리하고 원래 typed 원인을 보존한다. `Ready`나
`ActualCompute`를 만들거나 예약을 해제하여 정상 물리 완료로 표시하지 않는다.
이미 GPU 작업을 시작한 뒤의 논리 취소도 실제 완료 전에 buffer 재사용을 허용하지 않는다.
입력 검증 단계에서 실행 전 거부한 오류와 실제 native 실행 후 오류를 구별한다.

## 실제 UCI 연결의 고정 계약

다음 인수는 실제 Rules 상태 → C CUDA owner → D scheduler → B UCI 탐색의 연결이다.
`rz-uci`의 `onnx-cuda` feature와 명시적인 `--onnx-cuda` 실행 옵션이 함께 필요하며,
기존 `onnx-cpu` 단독 실행의 provider·backend identity·V1 인수 기록은 유지한다.
CUDA bundle 경로와 manifest raw SHA-256을 별도 옵션으로 받고, manifest의 ORT core를
기존 runtime 파일 pin과 대조한다. manifest 파일 hash와 canonical bundle digest는
서로 다른 근거다. 지원하지 않는 build·provider·프로필은 실행 전에 거부한다.

첫 native UCI CUDA 프로필은 Linux, device `0`, FP32, TF32 off, batch `1`, intra thread
`1`, search worker `1`, fresh/full step `1`, HistoryFill `No`, ORT arena `1 GiB`다.
C owner는 실제로 로드한 backend·encoding·action map·모델 identity, warm-up에서
관측한 CUDA kernel과 profile digest, 19개 native 파일의 실제 매핑을 확인한다.
주입한 mock worker는 자원·실패를 재현할 수 있지만 `CudaOnnx` origin을 얻지 않는다.

실행당 D 추가 예약은 host overhead `4096` byte, device `1 GiB`, pinned `0`이다.
요청의 device byte에는 이 overhead를 중복 기입하지 않는다. 세션 상주 admission
`1 GiB`는 bootstrap의 전체 예산에서 별도로 보존하며 실행 예약과 구분한다.
이 선언과 ORT arena 한도는 실제 VRAM 측정값이나 장치 전체의 강제 상한이 아니다.
실행 예약은 C가 물리 완료를 확인하기 전까지 보존하고, 불확실한 CUDA 실패는 원래
원인·세션·입력·예약을 격리한 상태로 남긴다.

CUDA startup/termination은 별도 V1 namespace의
`native-cuda-startup.v1.json`과 `native-cuda-termination.v1.json`으로 보존한다.
source·model·export·runtime·bundle·backend·encoding·action map·history·device·arena·
placement 근거를 잠그고 실제 CUDA origin과 물리 drain을 검증한다. GPU 결과를 기존
`actual_cpu_inference_observed` 필드나 `CpuOnnx` origin으로 표현하지 않는다.
E는 이 GPU 형식을 직접 검증하는 별도 feature·실행 명세를 사용한다. 19개 입력과
새 프로세스별 runtime 사본을 기존 CPU 입력·출력 예산에 억지로 맞추지 않으며,
추가 snapshot·사본·로그·시간 예산을 실행 전에 유한하게 확정한다.

요청의 physical drain과 process worker의 종료는 별도 사건이다. per-root 종료는
다음 root가 같은 session을 재사용할 수 있도록 요청만 drain한다. process 최종
종료에서는 정상 admission을 닫고 기존 lease·진단을 회수한 뒤, 캡처한 backend의
session destructor와 worker thread join까지 확인한다. 작은 reaper가 실제 join을
맡고 호출자는 결과 채널을 비차단 조회하므로 native destructor나 thread-local
cleanup이 caller deadline을 무한히 막지 않는다. 전체 final collection·join·재회수는
하나의 기존 2초 deadline을 공유한다. Pending·timeout·panic·quarantine을 성공으로
바꾸지 않고 원래 원인·owner·evidence를 보존한다. join 성공도 quarantine된 실행의
물리 완료를 증명하지 않으며, process에 고정된 native library의 unload를 뜻하지 않는다.

CUDA E 인수는 native 성공 기록과 외부 프로세스 종료를 함께 확인한다. 검증한 네
startup PID 각각에 대해 고정 Fastchess의 종료 TRACE가 한 번 존재하고 raw wait
status가 정확히 0이어야 한다. 누락·중복·다른 PID·renderer 변경·비정상 상태·강제
종료는 provider 인수 실패다. 감독자가 소유한 bounded stdout과 기존 artifact digest를
사용하며 엔진이 `[Engine]` 출력에 인용한 TRACE는 근거로 받지 않는다. 러너 자체의
exit 0만으로 각 엔진의 성공 종료를 추정하지 않는다. CPU V1의 필드 집합은 유지한다.

D의 `Finished`는 mailbox 적재, `PhysicalCompleted`는 owner의 완료 관측이다.
common evaluator의 최종 `poll` 검증과 B의 최종 scope·clock 검사를 통과한 소비는
각각 별도 기록이며 root 초기화와 실제 non-root backup도 구분한다. 관측 ring의
유실·counter overflow를 보고하고, 예약이 변한 경우에만 예약 snapshot을 적재한다.
이 연결 관측만으로 source ORT 시작·완료 시각이나 GPU 전송·kernel 구간을 주장하지
않는다. D02의 실제 source clock·종단 journal은 추가 인수로 남긴다.

## 실행 증거와 남은 인수

실제 검증에는 source SHA·계약 revision·binary SHA·전체 bundle·원본/ONNX/참조 hash,
장비·driver·OS·시계·자원 상한·명령·stdout/stderr·실제 CUDA profile·오류를 보존한다.
기존 참조 입력을 재사용하면 입력과 설정의 일치 근거를 기록하며 새 GPU 실행과 구분한다.
외부 프로세스의 유한 wall time과 자식 종료도 관리한다. CPU 테스트와 CI는 GPU 실험을
대신하지 않는다.

`maia_check`의 수치·provider 검사는 C03의 일부 인수다. 실제 A 상태와 D scheduler를
거친 GPU 요청, B의 유효 소비와 backup, 취소·deadline·root 교체의 GPU 수명, peak VRAM,
D02 종단 계측, D03 효과, 정식 paired 대국은 각각 별도 증거가 필요하다. 이 접점을
제공했다는 사실만으로 C03 전체·G2·GPU 성능·LC0 대비 강도 향상을 완료로 보고하지 않는다.

첫 실제 목표 장치 검사는 `bde687c7f269af3c7cd501201edde14c5c8ef642`에서 통과했다.
선정한 19개 native bundle과 Maia/FP32를 고정하여 CUDA 98 kernel 이벤트, 12개 독립
원본 참조와 batch 1/2/4/8/16, C worker와 같은 binary의 CPU 회귀를 확인했다.
이 결과는 수치·provider probe의 부분 인수다. source CI·profile·receipt·자원 상한·
관측값과 선행 실패, 남은 실제 A/D GPU 연결은 [통합 인수 상태](INTEGRATION-STATUS.md)에 둔다.
