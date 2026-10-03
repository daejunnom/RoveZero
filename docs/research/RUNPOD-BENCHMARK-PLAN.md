# Runpod 외부 GPU 벤치 준비 계획

작성일: 2026-10-03. 상태: **CPU 사전 최적화 검증 / Community 후보 확인 / Pod 미생성**.

## 근거와 현재 범위

- 총괄이 확인한 PR #14 병합 기준은 `develop`의
  `164a898061e058f57519b9aed2c45294796696ba`, 최소 공통 계약은 `0.1`이다.
- 기존 제품 인수 checkpoint는 `e45dd1050d8782ff4ae3e524341cdbac52209cf5`다. 후속 문서 head와
  제품 소스·실행 binary를 구분하고, 새 실행 전 실제 source SHA와 binary hash를 잠근다.
- 이 문서는 공식 Runpod 조사와 총괄의 일반 배포 폼 관측을 정리한다.
  Pod 생성·유료 API·GPU 실행·계정 설정·Credentials 접근·MCP 설치를 수행하지 않았다.
- 후속 사용자 지시는 **GPU A/B 전에 의미 보존 최적화를 먼저 검증**, **network volume 미사용**,
  **Community GPU 후보 확인**, **초기 총 지출 가능액 약 US$200**다.
  [사전 최적화 기록](PRE-RUNPOD-OPTIMIZATION.md)의 제품 소스는
  `1cdd707922dcaaf234b4cd162210ce9f2fa512bb`이며 새 장비의 수치·종단 검사는 남아 있다.
- 이전의 GPU 미선정 이력을 유지한다. 현재 Community 4090을 우선 후보로 제안하며
  GPU 최종 선택·quote·유한 실행 명세·접속 설정을 잠근 뒤 실행한다.
- 기존 RTX 4050 6GB의 CUDA 수치·UCI·pair smoke 증거와 실패 이력을 보존한다.
  외부 GPU 결과는 새 환경 증거로 추가하며 기존 장비의 결과를 덮어쓰지 않는다.
- 기존 6-ply pair smoke의 `strength_eligible=false`를 유지한다. 외부 자원 선택을
  D02 완료·LC0 강도 향상·모델 승격·학습 성공으로 보고하지 않는다.
- 프로젝트의 [작업 규약](../../AGENTS.md), [공통 계약](../CONTRACTS.md),
  [평가 절차](../EVALUATION-PROTOCOL.md), [가중치 선정](../WEIGHT-SELECTION.md)을 따른다.

## 실행 상품과 장비 후보

첫 실행안은 **Community Cloud / on-demand / 단일 NVIDIA GPU Pod**다.
이전 Secure Cloud 제안은 사용자 후속 지시로 대체했다. 자료의 기밀성 때문에 Secure를
필수 조건으로 두지 않지만, loaded origin·입력 identity·자원 한도·실패 보존은 계속 검증한다.
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

| 현재 Community 후보 / 선택 상태 | GPU 표시 가격 | VRAM / RAM / vCPU | Available 표시 |
|---|---:|---|---|
| RTX 4090 / 우선 후보, 미배포 | $0.34/h | 24GB / 125GB / 25 | 2 max |
| RTX A5000 / 비용 비교 후보 | $0.16/h | 24GB / 25GB / 3 | 1 max |
| L40S / VRAM 비교 후보 | $0.79/h | 48GB / 251GB / 24 | 4 max |

위 값은 로그인된 일반 deploy 폼을 **2026-10-03 03:17 UTC**에 읽은 관측이며
확정 견적·배포 성공·실제 할당 사양이 아니다. `max`를 재고 총수로 해석하지 않는다.
조건은 Community, Any region, Available, 최소 vCPU 1·RAM/VRAM 0,
인터넷 속도 400·CUDA Any·public IP 필수 아님이다. 템플릿은 아래 PyTorch 후보다.
4090/A5000은 최초 Community 조회에서는 없었고 후속 조회에 나타났다. 가용성은 변한다.
이전 Secure 관측은 4090 $0.74/h·24/31/12, A5000 $0.27/h·24/50/9였으며
현재 Community 가격과 섞지 않는다. 최종 quote에는 disk·기타 비용도 포함한다.
GPU 종류·개수·최소 RAM/vCPU·datacenter 조건은 지정할 수 있으나 배포 성공은 별도 확인한다.
[공개 가격표](https://www.runpod.io/pricing),
[Pod 배포 조건](https://docs.runpod.io/api-reference-v2/pods/create-a-pod)

현재 4090은 A5000보다 vCPU/RAM 여유가 있어 CPU 준비·native bootstrap을 함께 보는
첫 외부 기준 후보로 적합하다. 이는 offer 사양에 따른 제안이며 실측 속도 비교가 아니다.
선정 후 재고 부족을 이유로 다른 GPU나 CPU로 자동 대체하지 않는다.
RTX 4050과 다른 GPU의 시간·처리량을 직접 합치지 않고 내부 A/B는 같은 장비에서 수행한다.
arena 크기를 제한하는 것만으로 외부 GPU를 RTX 4050과 동등한 장비로 만들 수 없다.

이 조회는 신규 `/deploy` 폼이다. 기존 리소스 부재·Pod 생성·접속·GPU 사용 가능 상태를
확인한 결과로 확대하지 않는다. 이전에 관측한 landing도 기존 리소스 목록 증거가 아니다.

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
Pod의 SSH 인증·파일 전송 가능 여부는 미확인이다. Oracle 저장 서버의 SSH 접속과
작은 archive의 SCP 왕복은 확인했다. Pod→Oracle 경로는 새 Pod에서 별도로 검사한다.
public IP 없는 Pod에서도 proxy SSH/Web terminal로 명령을 실행하고 outbound SSH로
Oracle에 전송하는 안을 우선한다. inbound SCP가 꼭 필요하면 full SSH 조건을 따로 잠근다.
인증 자료와 **Pod Env 등록은 사용자 담당**이다. 에이전트는 지정한 key를 SSH 인증
인자로만 사용하고 내용을 읽거나 출력·hash·보고서·image·Git에 포함하지 않는다.
새 key·port·접근 권한을 임의로 추가하지 않는다.
[접속 방법](https://docs.runpod.io/pods/connect-to-a-pod),
[SSH와 전송 지원](https://docs.runpod.io/pods/configuration/use-ssh)

**network volume은 생성·연결하지 않는다.** 첫 smoke는 로컬 container disk에서 실행하고,
보존할 결과는 Oracle로 회수한다. 첫 disk 크기는 **80GB 제안**이며 image·빌드·입력·
bounded attempt의 실제 요구량과 최종 quote로 확정한다. 80GB의 공개 월 단가 환산은
약 $0.011/h이고 현재 4090 카드와 합친 산술 예시는 약 $0.351/h다. 최종 견적이 아니다.
container disk는 stop/restart 때 지워진다. Pod volume disk를 쓰는 후속 선택이 생기더라도
terminate 때 지워지므로 외부 회수를 대신하지 않는다. network/global volume으로 조용히
대체하지 않는다.
[스토리지 수명](https://docs.runpod.io/pods/storage/types),
[Network volume 조건](https://docs.runpod.io/storage/network-volumes)

원시 profile·journal·PGN·로그·모델·native copy는 저장소 밖의 작업 전용 output root에 둔다.
root manifest에 논리 경로·길이·SHA·source/환경/자원·실패·회수 상태를 기록한다.
회수 후 manifest와 실제 작은 인수 파일의 hash를 대조하고 필요한 원시 증거의 회수를 확인한
뒤 승인된 Pod terminate를 수행한다. 개인 계정·호스트 식별자·credentials는 공유 문서에 넣지 않는다.
stop은 GPU를 반환하며 storage 비용은 남고, restart 시 GPU 재고가 달라질 수 있다.
[Pod 수명·회수](https://docs.runpod.io/pods/manage-pods),
[중지 후 GPU 가용성](https://docs.runpod.io/pods/troubleshooting/zero-gpus)

Oracle의 전용 저장 공간은 논리 이름 `oracle-artifacts/coordinator-20261003`으로 기록한다.
서버 주소·개인 key 경로는 공유 문서에 넣지 않는다. 확인된 가용 공간은 약 **31.7GB**다.
첫 보존 admission은 repository 1GiB·archive 64MiB이며 OS quota를 설정한 것은 아니다.
D/E/F 원본 719개와 과거 E 스크립트 2개를 하나의 435,710-byte archive로 보존했다.
721개 모두 원래 receipt의 Git blob·SHA-256·길이와 대조했고 archive SCP 왕복 hash도 확인했다.
이 스크립트들을 새 환경에서 실행한 것은 아니다. 상세 digest는 사전 최적화 기록에 둔다.

입력/native 원본은 hash별 한 사본을 재사용할 수 있지만 실행마다 필요한 readonly pin·
별도 inode·물리 수명 검증을 없애지 않는다. 성공/실패 attempt의 raw journal·PGN·receipt는
각각 회수하고, 동일 witness나 재생성 가능한 큰 private copy는 manifest의 논리 참조와
원본 digest를 통해 중복 전송을 줄인다. 원시 증거 자체를 생략하거나 무승부/성공으로 바꾸지 않는다.
31.7GB에 16GiB attempt 여러 개와 native copies를 무제한 보존할 수 있다고 가정하지 않는다.
유한 stage마다 회수·원격 bytes/hash 확인·보존 완료 receipt를 발급한다. 저장 용량·회수 실패는
phase 실패로 보존하며 미회수 결과를 둔 채 Pod를 종료하지 않는다.

## 유한 실행 순서와 남은 결정

1. **코드 사전 검사:** 의미 보존 최적화·같은 입력의 전후 CPU 측정·동일 상태 witness·
   전체 CPU CI를 인수한다. 이 단계의 비용 감소를 GPU/D03 성공으로 승격하지 않는다.
   이번 source의 CPU 사전 검사는 사전 최적화 기록을 따른다.
2. **잠금·preflight:** 단일 source/image/binary/input/bundle과 실제 GPU·driver·process VRAM
   계측·cgroup capability·wall/resource/output·취소·회수·과금 중지 경계를 확정한다.
3. **1회 bounded smoke:** 새 환경에서 CUDA parity → UCI → 6-ply pair 연결을 확인한다.
   CPU regression은 별도 결과로 남긴다. startup/termination·actual CUDA profile·whole mapping·
   physical drain·guarded 소비·실패를 보존하고 이를 강도나 전체 D02 인수로 확대하지 않는다.
4. **D02:** 동일 장비·source/model/backend/input hash·자원 예산에서 encoding·queue·전송·
   실제 GPU 완료·전체 자기 wallclock의 full journal과 P50/P95/P99를 유한 표본으로 측정한다.
   cold/warm을 분리하고 반복 수·입력 순서·중단 조건을 사전에 잠근다.
5. **내부 paired A/B:** 같은 장비·W/S·hash·입력·budget을 유지하고 runtime 한 변수만 바꾼다.
   LC0 대조는 별도 commit/weights/backend/options와 같은 장비·시계·opening·색 교대 계약으로
   수행한다. 소형 smoke나 처리량 개선을 LC0 강도 향상으로 승격하지 않는다.

초기 총 지출 한도는 사용자 지정 **약 US$200**다. 총괄의 첫 하위 실행 한도는
**US$5 / 전체 60분 / 단일 Pod·단일 attempt**로 두며 설치·idle·회수·disk 비용을 포함한다.
그다음 D02·첫 A/B 묶음은 추가 US$20·전체 8시간 이내의 계획으로 두고, 실행할 입력·
반복·cold/warm·실패/중단 조건을 잠근다. 나머지 약 US$175는 후속 연구 여유분이며 자동 소비하지 않는다.
각 단계에서 실제 누적 비용·잔여 예산·quote를 재확인하고 더 낮은 비용/시간 한도부터 종료한다.
견적 불명·세금/추가비용 누락·외부 lifecycle controller 미검증이면 유료 실행을 시작하지 않는다.
이 하위 한도는 초기 총예산 안의 도입 기본값이며 총예산 증액을 요구하지 않는다.
총괄이 담당들의 관련 GPU·disk·전송 지출을 합산한다. 각 에이전트에게 별도의 US$200나
새 Pod 실행 권한을 부여하는 규약으로 해석하지 않는다. 이번 Oracle 작업은 기존 서버의
지정 저장 공간만 사용했으며 새 instance·volume·유료 서비스·계정 설정을 생성하지 않았다.

이전 `US$5 / 60분 / 50GB network volume` 예시는 폐기하고 network volume 미사용으로
대체했다. 입력 약 2.98GB·attempt 상한 16GiB·기존 약 14.86GB private-copy 사례·image/
빌드 요구량을 disk admission에서 함께 계산한다. 유한 보존 기간은 초기 자동 삭제 없이
해당 연구의 인수·회수 확인까지로 두며, 이후 정리는 소유 범위·원본 보존을 확인해 별도로 진행한다.
남은 것은 GPU 최종 선택, 사용자 Pod Env/접속 준비, image digest/실제 할당·cgroup와
계측 capability, 과금 종료 controller와 한 회의 잠금 명세다. 이번 작업에서 유료 Pod를
생성하거나 GPU A/B를 실행하지 않았다.
