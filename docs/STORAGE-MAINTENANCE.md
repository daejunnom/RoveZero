# 생성물 수명과 명시적 유지보수

관리 기능은 새로 등록한 소유 영역을 제한한다. Windows C: 전체, WSL VHD의
물리 할당량 또는 과거에 직접 만든 모든 디렉터리를 제한하는 기능은 아니다.
빌드 cache, 보존 결과, 원 runtime 자산과 실제 loader cache는 서로 다른 책임이다.

## 고정 빌드 슬롯

일반 실행은 `python scripts/run_managed.py -- cargo ...`를 사용한다. 별도 진단이
필요하면 `--purpose startup-caller`처럼 **안정적인 목적 이름**을 지정한다.
시각·SHA·PID마다 target 형제 폴더를 만들지 않는다. checkout과 purpose가 함께
슬롯을 식별하며 기본 슬롯도 같은 카탈로그의 8GiB/12GiB/최대 4개 계산에 들어간다.
`CARGO_TARGET_DIR`로 관리 영역 밖을 지정하면 기존처럼 거부한다.

직접 Cargo를 실행하거나 임의 경로에 복사한 과거 target에는 관리 코드의 소유권이
없다. 아래 inventory로 누락을 확인할 수 있지만, 이름이나 오래된 시각을 근거로
자동 채택·삭제하지 않는다. 새 진단은 위 접점으로 옮겨 재발을 막는다.

## 출력 예약과 보존

`managed_outputs.py`는 모델 검증과 followup 실행기의 새 출력에 공통 카탈로그를
연결한다. 모델 검증은 기본 8GiB를 예약하고, followup은 기존 출력 상한을 예약한다.
등록 결과 전체의 기본 상한은 18GiB이며 최대 1,024개 기록을 허용한다. 완료·실패
결과도 합계에 포함한다. 여유가 부족하면 새 실행을 거부한다.

카탈로그 위치는 Windows `%APPDATA%/RoveZero/storage/outputs-v1`, Linux
`${XDG_CACHE_HOME:-$HOME/.cache}/rovezero/storage/outputs-v1`, CI
`$RUNNER_TEMP/RoveZero/storage/outputs-v1`다. 등록한 실제 출력은 원래의
reports/runs 경로를 유지하며, 경로·inode·소유 token을 카탈로그에서 대조한다.
한 카탈로그의 producer는 OS gate로 직렬화한다. Rust가 직접 생성해야 하는
fresh 경로는 deferred 예약으로 남겨 CLI 계약을 보존한다.

- 14일의 검토 기한은 **회수·보존 결정을 검토할 시점**이다. 만료가 모델·PGN·원시
  로그·실패 결과의 삭제 권한이 되지 않는다. 결과가 한도를 채우면 먼저 보존 근거와
  원래 native retention journal을 확인하고 별도 유지보수를 진행한다.
- 카탈로그는 과거 reports/runs를 자동으로 순회·등록하지 않는다. 따라서 초기 도입
  시 기존 미등록 자료까지 18GiB 안으로 줄였다는 주장은 하지 않는다.
- 실행 중 run_managed의 5초 관측과 producer의 검사, 종료 시 최종 크기 검사를
  함께 사용한다. 관측 사이 증가는 가능하며 kernel quota나 물리 사용량 제한은 아니다.
- `tree_gone`이 True가 아니면 active 예약을 유지하여 다음 producer를 차단한다.
  모델 검증의 venv는 바깥 관리 scratch에 남기고, 원 감독자가 실제 자식 트리 종료를
  확인한 뒤에만 임시 자료를 회수하고 같은 owner에 묶인 출력을 확정한다.
- OS 프로세스 종료는 GPU 물리 fence를 대신하지 않는다. 입력·seed·workspace와
  provider 완료 unknown의 정리는 기존 native 수명·quarantine·retention 계약을 따른다.

CPU 모델 검증의 원 ORT DSO는 실행별 native-runtime 대신 digest별 공유 저장소를
기본으로 사용한다. 기본 위치는 관리 루트의 `cache/pals-cpu-runtime`이며 4GiB
합계를 검사한다. 기존 `--runtime-store` 지정도 유지한다. 읽기는 크기 제한과
streaming hash를 사용하고, 손상된 기존 entry는 보존·거부한다. 이는 Rust loader가
관리하는 runtime cache 또는 GPU 버퍼를 합치거나 삭제하는 변경이 아니다.

CI native bundle은 원 FD→private ELF→tar의 custody를 먼저 검증한다. 성공과 모든
command tree 종료가 확인된 뒤 target·cargo-home·tmp·private ELF만 정리한다.
원시 logs, artifact와 build receipt는 남긴다. `compile-cleanup.json`의 최종 상태는
build 성공 영수증과 별도로 확인하며 실패는 `cleanup-failure.json`과 exit에 남긴다.

## 중단 lease 복구

`owner.json`을 오래됐다는 이유로 삭제하지 않는다. 새 실행은 host-local 식별,
boot, process 생성 시점, root inode를 원 lease에 묶는다. 원 supervisor가 tree
종료를 확인하면 정리보다 먼저 token에 묶인 `closure.json`을 저장한다. 일반
실행과 유지보수는 같은 OS gate와 catalog 디렉터리 identity를 확인한다.

`storage_maintenance.py recover-build`는 **원 owner의 raw SHA256**을 요구하고,
다음 둘 중 하나가 증명될 때만 그 소유 슬롯의 재생성 target/tmp를 회수한다.

1. 같은 host의 확인된 재부팅으로 이전 boot의 프로세스가 남을 수 없음.
2. 원 lease와 정확히 같은 종료 증거가 있고 원 owner가 종료됨. PID 재사용은
   process 생성 시점으로 구분하며, 다른 PID를 signal하지 않는다.

hash와 parse는 같은 held FD 바이트에서 얻고 commit 직전 다시 대조한다. root·slot·
gate가 바뀌거나 기록이 불명확하면 보존한다. 구형 v1 owner에는 위 근거가 없으므로
새 복구 명령이 임의로 권한을 만들어 주지 않는다. 별도 원 트리 조사 대상으로 남긴다.

`recover-output`은 같은 host의 재부팅, 원 catalog record의 SHA·경로·inode를
확인하여 interrupted-retained로 옮긴다. 이 명령은 결과나 GPU 입력을 삭제하지
않는다. 생성 중이던 deferred 경로의 identity가 확정되지 않았다면 계속 거부한다.

## 명령 예시와 관측 한계

다음은 저장소 루트에서 실행하는 예시다. 복구 명령은 실제 대조한 원 digest를
전달해야 하며 예시 값을 사용하지 않는다. CLI는 새 lease를 자동으로 훔치지 않는다.

```powershell
python scripts/run_managed.py --purpose startup-caller -- cargo build --locked -p rz-uci
python scripts/storage_maintenance.py inventory --root "$env:APPDATA/RoveZero/build" --root "$env:APPDATA/RoveZero/reports" --root "$env:APPDATA/RoveZero/runs"
python scripts/storage_maintenance.py recover-build --root "$env:APPDATA/RoveZero/build/managed-v1" --owner-sha256 <original-owner-sha256>
```

inventory는 최대 16개의 **명시적·겹치지 않는 root**만 길이 기준으로 조사하고 삭제하지
않는다. root별 비밀 이름·link 경계·entry 수를 검사한다. Linux root 사용자, 일반
사용자, `/var/tmp`와 Windows는 필요한 root를 각각 명시하여 따로 관측한다.
서로 다른 root의 hardlink 중복과 VHD 안팎의 물리 사용량은 합계에서 해결하지 않는다.
파일 삭제와 VHD 축소는 별개이며 전역 WSL 종료·압축을 자동 실행하지 않는다.

등록 카탈로그 밖의 임의 스크립트·수동 복사는 이 한도를 우회할 수 있다. 새 대형
producer는 `OutputLease` 또는 기존 native retention 접점에 연결해야 한다.
기존 미등록 자료의 분류·회수는 실제 소유자·사용 여부·보존 근거 확인 후의 별도 작업이다.
