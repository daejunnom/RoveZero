# Runpod 외부 GPU 벤치 준비 계획

작성일: 2026-10-03. 상태: **조사·준비 계획 / GPU 선택 및 유료 실행 미승인**.

## 근거와 현재 범위

- 총괄이 확인한 PR #14 병합 기준은 `develop`의
  `164a898061e058f57519b9aed2c45294796696ba`, 최소 공통 계약은 `0.1`이다.
- 기존 제품 인수 checkpoint는 `e45dd1050d8782ff4ae3e524341cdbac52209cf5`다. 후속 문서 head와
  제품 소스·실행 binary를 구분하고, 새 실행 전 실제 source SHA와 binary hash를 잠근다.
- 이 문서는 공식 Runpod 조사와 총괄의 일반 배포 폼 관측을 정리한다.
  Pod 생성·유료 API·GPU 실행·계정 설정·Credentials 접근·MCP 설치를 수행하지 않았다.
- 사용자는 후속 GPU benchmark를 로컬 밖에서 수행할 예정이며,
  **이번 응답 이후 GPU를 지정한다.** 아래 두 후보 중 선택된 GPU는 아직 없다.
- 기존 RTX 4050 6GB의 CUDA 수치·UCI·pair smoke 증거와 실패 이력을 보존한다.
  외부 GPU 결과는 새 환경 증거로 추가하며 기존 장비의 결과를 덮어쓰지 않는다.
- 기존 6-ply pair smoke의 `strength_eligible=false`를 유지한다. 외부 자원 선택을
  D02 완료·LC0 강도 향상·모델 승격·학습 성공으로 보고하지 않는다.
- 프로젝트의 [작업 규약](../../AGENTS.md), [공통 계약](../CONTRACTS.md),
  [평가 절차](../EVALUATION-PROTOCOL.md), [가중치 선정](../WEIGHT-SELECTION.md)을 따른다.

## 실행 상품과 장비 후보

첫 실행안은 **Secure Cloud / on-demand / 단일 NVIDIA GPU Pod**다.
Pods는 컨테이너·프로세스·저장소를 직접 관리하는 환경을 제공한다.
[Pods 개요](https://docs.runpod.io/pods/overview),
[On-demand 과금](https://docs.runpod.io/pods/pricing)

Serverless는 custom handler와 endpoint, 자동 worker 수명·확장을 추가한다.
GPU 우선순위도 설정할 수 있지만 여러 종류를 허용하면 재고에 따라 장비가 달라질 수 있다.
시작·실행·idle timeout 동안 과금하며 active worker는 대기 중에도 과금한다.
현재 UCI·runtime·D02의 직접 실행에는 Pod를 제안하고 Serverless 도입은 별도 연구로 둔다.
[Serverless 개요](https://docs.runpod.io/serverless/overview),
[Endpoint 설정](https://docs.runpod.io/serverless/endpoints/endpoint-configurations),
[Serverless 과금](https://docs.runpod.io/serverless/pricing)

| 후보 / 선택 상태 | 공개 가격표 | 공개 표의 VRAM / RAM / vCPU | 일반 deploy 폼의 공개 offer 구성: 총괄 관측 |
|---|---:|---|---|
| RTX 4090 / 미선택 | $0.74/h | 24GB / 41GB / 6 | $0.74/h, 24GB / 31GB / 12 |
| RTX A5000 / 미선택 | $0.27/h | 24GB / 25GB / 9 | 최초 $0.27/h, 24GB / 50GB / 9; 후속 조회에서 표시 없음 |

위 값은 2026-10-03 확인 기준이며 확정 견적·실행 장비 사양이 아니다.
공개 가격과 실제 offer의 CPU/RAM 차이를 유지하고 배포 시 최종 quote와 실제 할당을 기록한다.
A5000은 최초 조회 뒤 후속 deploy 조회에서 표시되지 않았다. 재고·표시가 변동하는 후보이며
현재 available을 보장하지 않는다. 특정 지역·필터·GPU를 선택하지 않은 관측이다.
GPU 종류·개수·최소 RAM/vCPU·datacenter 조건은 지정할 수 있으나 배포 성공은 별도 확인한다.
[공개 가격표](https://www.runpod.io/pricing),
[Pod 배포 조건](https://docs.runpod.io/api-reference-v2/pods/create-a-pod)

4090을 외부 기준 장비로 삼는 안과 A5000으로 첫 smoke 비용을 줄이는 안이 있다.
선정 후 재고 부족을 이유로 다른 GPU나 CPU로 자동 대체하지 않는다.
RTX 4050과 다른 GPU의 시간·처리량을 직접 합치지 않고 내부 A/B는 같은 장비에서 수행한다.
arena 크기를 제한하는 것만으로 외부 GPU를 RTX 4050과 동등한 장비로 만들 수 없다.

총괄의 Pods 이동은 신규 `/deploy` 폼으로 이어졌고 Storage에서는 안내 landing을 관측했다.
이는 기존 Pod·volume 목록을 확인한 결과가 아니므로 **기존 리소스 없음·volume 선택 완료**로
판정하지 않는다. live 재고·지역·최종 quote·계정 실행 capability도 아직 미확정이다.

## 컨테이너·입력·native 재현성

- 총괄이 관측한 기본 템플릿은 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`다.
  이 tag는 후보이며 검증된 실행 image가 아니다. 실행 전 image digest·OS/ABI를 고정한다.
- root 소유 toolchain·workspace lock·source SHA·feature·release binary hash를 잠그고
  변경 없이 옮긴 코드와 새 환경에서 다시 실행한 검사를 구분한다.
- **ORT 1.22.0**과 host driver/CUDA 조건을 구분한다. CUDA compatibility 필터는 host 선택
  조건이며 정확한 driver 버전 고정 보장으로 취급하지 않는다. 실제 driver·GPU UUID·VRAM,
  CPU 모델·할당 vCPU/RAM·datacenter를 관측하고 환경 fingerprint를 남긴다.
- 기존 closed bundle **19파일: NVIDIA 16 + ORT 3**의 이름·길이·SHA·bundle digest를 고정한다.
  이미지의 같은 이름 라이브러리나 다른 wheel로 조용히 바꾸지 않는다.
  host libc·동적 loader·driver 같은 플랫폼 의존성은 이 19파일의 검증 범위와 구분해
  환경·ABI 근거로 기록한다. bundle 일치만으로 컨테이너 전체의 의존성 고정을 주장하지 않는다.
- native 파일은 소유한 private readonly copy로 검증한다. loader 입력의 identity·수명·실패
  보존을 유지하고 preload 전 검사, 실제 loaded origin, 전체 runtime mapping 감사를 수행한다.
- ambient loader 변수 제거와 fresh Rust 프로세스 경계를 유지한다. 다른 native library가
  이미 resident인 프로세스를 사용하거나 GPU 실패를 CPU 성공으로 바꾸지 않는다.
- 첫 실제 CUDA 조건은 B1·device0·arena 1GiB·FP32·TF32 off·worker1·fresh다.
  arena admission 선언을 측정 VRAM peak 또는 hard cap으로 표현하지 않는다.
- source weights·protobuf·ONNX·export manifest·encoding/action/history·backend identity를
  잠그고 새 장비에서 수치와 placement를 확인한다. 기존 host 결과의 자동 재사용은 금지한다.
- Maia의 기존 권리·출처·hash와 사용 범위를 유지하고 클라우드 복제·공개 image 포함 범위를
  [가중치 선정 기록](../WEIGHT-SELECTION.md)으로 확인한다. T70 보류는 해제하지 않는다.

custom image/template으로 환경을 재사용할 수 있으며 host CUDA compatibility도 선택할 수 있다.
위의 구체 hash·loader·모델 인수 조건은 RoveZero 계약이며 Runpod의 제공 보장이 아니다.
[Custom template](https://docs.runpod.io/pods/templates/create-custom-template),
[Host CUDA 조건](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)

## 플랫폼 제어와 실행 한계

- 기존 외부 helper의 `/usr/lib/wsl/lib/nvidia-smi` 경로는 WSL 전용이다.
  새 Linux 환경의 실제 도구 경로·기능을 검증하는 bounded platform adapter가 필요하다.
- 기존 helper는 `/sys/fs/cgroup`에 owned group을 생성하고 RAM·CPU·pids를 제어한다.
  Pod 컨테이너의 cgroup v2 위임·쓰기 권한·enforcement를 실제로 확인하기 전 재사용 성공으로
  표시하지 않는다. 위임 실패를 privileged 실행·host 설정 변경으로 임의 우회하지 않는다.
- cgroup 제어가 없으면 기존 UCI RAM 4GiB / pair RAM 8GiB·CPU 2코어·pids128의
  aggregate hard cap을 주장할 수 없다.
  유한 child timeout·RSS 관측/watchdog·소유 프로세스 종료는 별도 수단이며 같은 증거가 아니다.
- RLIMIT_AS 128GiB는 process별 가상 주소 공간 제한이다. RAM/VRAM 128GiB를 요구한다는
  뜻이 아니며 aggregate RAM 제한을 대신하지도 않는다. 실제 적용 여부를 receipt에 기록한다.
- 한도 미지원이면 새 자원 명세를 명시적으로 확정하거나 해당 gate를 미실행으로 둔다.
  OOM·지원 실패 후 batch·정밀도·GPU·장비·예산을 조용히 바꾸거나 무한 재시도하지 않는다.
- benchmark deadline과 Pod 과금 중지는 다르다. 프로세스 timeout 외에 별도 외부 lifecycle
  controller와 중지/종료 상태·잔여 프로세스 확인이 필요하다. 검증되지 않은 자동 종료 flag를
  명령에 넣지 않는다. 이번 계획에는 controller 구현·실행 완료 증거가 없다.

## 접속·보존·회수

SSH를 실행·회수의 기본안으로, Web terminal을 짧은 확인용으로 제안한다.
basic proxy SSH에는 SCP/SFTP가 없고 public IP를 지원하는 full SSH에는 있다.
JupyterLab은 템플릿·port 설정에 의존하며 Rust 벤치의 필수 조건이 아니다.
현재 계정의 SSH 인증·파일 전송 가능 여부는 미확인이다. 실제 연결은 사용자가 지정한
기존 접속 수단으로 준비하고 새 key·public port·접근 권한을 임의로 추가하지 않는다.
[접속 방법](https://docs.runpod.io/pods/connect-to-a-pod),
[SSH와 전송 지원](https://docs.runpod.io/pods/configuration/use-ssh)

container disk는 stop/restart 때 삭제되고, Pod volume disk는 stop 후 유지되지만 terminate 때
삭제된다. network volume은 Pod와 독립적으로 유지되며 비용도 계속된다.
일반 network volume은 첫 1TB $0.07/GB/month이고 Secure Cloud Pod에서 위치에 맞춰 배포 시
연결한다. volume 위치가 GPU 가용성을 제한하므로 장비·지역을 함께 정한다.
[스토리지 수명](https://docs.runpod.io/pods/storage/types),
[Network volume 조건](https://docs.runpod.io/storage/network-volumes)

원시 profile·journal·PGN·로그·모델·native copy는 저장소 밖의 작업 전용 output root에 둔다.
root manifest에 논리 경로·길이·SHA·source/환경/자원·실패·회수 상태를 기록한다.
회수 후 manifest와 실제 작은 인수 파일의 hash를 대조하고 필요한 원시 증거의 회수를 확인한
뒤 승인된 Pod terminate를 수행한다. 개인 계정·호스트 식별자·credentials는 공유 문서에 넣지 않는다.
stop은 GPU를 반환하며 storage 비용은 남고, restart 시 GPU 재고가 달라질 수 있다.
[Pod 수명·회수](https://docs.runpod.io/pods/manage-pods),
[중지 후 GPU 가용성](https://docs.runpod.io/pods/troubleshooting/zero-gpus)

## 유한 실행 순서와 남은 결정

1. **잠금·preflight:** 단일 source/image/binary/input/bundle과 실제 GPU·driver·process VRAM
   계측·cgroup capability·wall/resource/output·취소·회수·과금 중지 경계를 확정한다.
2. **1회 bounded smoke:** 새 환경에서 CUDA parity → UCI → 6-ply pair 연결을 확인한다.
   CPU regression은 별도 결과로 남긴다. startup/termination·actual CUDA profile·whole mapping·
   physical drain·guarded 소비·실패를 보존하고 이를 강도나 전체 D02 인수로 확대하지 않는다.
3. **D02:** 동일 장비·source/model/backend/input hash·자원 예산에서 encoding·queue·전송·
   실제 GPU 완료·전체 자기 wallclock의 full journal과 P50/P95/P99를 유한 표본으로 측정한다.
   cold/warm을 분리하고 반복 수·입력 순서·중단 조건을 사전에 잠근다.
4. **내부 paired A/B:** 같은 장비·W/S·hash·입력·budget을 유지하고 runtime 한 변수만 바꾼다.
   LC0 대조는 별도 commit/weights/backend/options와 같은 장비·시계·opening·색 교대 계약으로
   수행한다. 소형 smoke나 처리량 개선을 LC0 강도 향상으로 승격하지 않는다.

남은 최소 사용자 결정은 **GPU**, **최대 총 wall·총비용**, **storage 용량·지역·보존 기간**이다.
`US$5 / 60분 / 50GB network volume`은 첫 예시 제안이며 승인된 예산·기본값이 아니다.
50GB의 월 표시 storage 비용 $3.50은 산술 예시이며 최종 견적이 아니다.
20GB는 최소 권장량으로 삼지 않는다. 입력 약 2.98GB와 attempt 상한 16GiB(약 17.18GB)만
합쳐도 약 20.16GB이며, 총괄이 관측한 기존 run 약 14.86GB의 실패·성공 보존 사례도 고려한다.
native copy·입력·bounded output·보존 attempt 수의 실제 용량을 산정한 뒤 storage를 확정한다.
기존 volume 선택·리소스 부재·배포 가능 상태는 미확정이다. 이 결정과 구체 실행 명세가
정해지기 전 유료 Pod·volume·API를 생성하거나 실행하지 않는다.
