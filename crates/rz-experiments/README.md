# rz-experiments — E01 입력 명세

TASK-E01의 첫 구현이다. 버전이 있는 실행 입력을 검증·잠그고, 다시 읽을 때 입력
SHA-256과 정책을 재검증한다. E의 내부 직렬화 형식이며 총괄의 `rz-contracts`를
정의하거나 대체하지 않는다. 기준 `develop`에는 아직 Cargo workspace와 공통 타입이
없어 이 패키지를 독립적으로 빌드한다. 이후 공통 계약 0.1은
[PR #7](https://github.com/daejunnom/RoveZero/pull/7)에 게시됐으며 정식 통합은 I01에서 진행한다.
root Cargo·CI는 변경하지 않는다.

## 제공하는 동작

- `validate`: 필수 타입·단위·digest·source·CARD·history 표시·유한 예산·실패 정책 검사.
- `lock`: 검증된 입력에 canonical JSON SHA-256을 붙여 새 파일로 저장. 기존 파일은 보존.
- `verify`: 잠금 버전·입력 digest·정책 재검증. 명시한 artifact root와 byte 상한이 있으면
  목록의 파일 크기·SHA-256도 대조.

잠금 쓰기·동기화에 실패하면 오류를 반환하고 부분 출력은 보존한다. 자동 삭제로
경로가 교체된 파일을 지우지 않으며, 실패한 출력은 정상 잠금으로 사용하면 안 된다.

모든 입력 잠금은 **`execution_ready=false`**다. E02의 실제 시작 영수증·합법 상태
복원·PGN 감사·clock/resource/fairness 검증이 필요한 입력 명세다. 실행 진입점은
아직 제공하지 않는다. 실제 옵션 관측의 출처·내용 검증, bootstrap 계산과 paired
SPRT/GSPRT, 결과 영수증·pair/attempt 원장은 후속 작업이다. 통계 계획에 bootstrap을
기록할 수 있지만 E01은 통계 계산 성공이나 구현 인수의 증거를 만들지 않는다.

## 독립 빌드와 사용

저장소 루트에서 실행한다. 빌드·잠금 출력은 저장소 밖 본인 소유 디렉터리를 사용한다.
`ARTIFACT_ROOT`는 이 E 작업의 검증 산출물 루트다. 검증은 단기 로컬 실행이며,
계속 사용할 인수 자료는 임시 환경 종료 전에 보존 위치로 회수한다.

```sh
export ARTIFACT_ROOT=/path/outside/checkout/rovezero-e
export CARGO_TARGET_DIR="$ARTIFACT_ROOT/build"
mkdir -p "$ARTIFACT_ROOT/runs"
cargo check --manifest-path crates/rz-experiments/Cargo.toml
cargo test --manifest-path crates/rz-experiments/Cargo.toml --locked
cargo clippy --manifest-path crates/rz-experiments/Cargo.toml --all-targets --locked -- -D warnings
cargo fmt --manifest-path crates/rz-experiments/Cargo.toml -- --check

cargo run --manifest-path crates/rz-experiments/Cargo.toml -- validate experiments/baselines/fixtures/e01-input.json
cargo run --manifest-path crates/rz-experiments/Cargo.toml -- lock experiments/baselines/fixtures/e01-input.json "$ARTIFACT_ROOT/runs/e01-input-lock.json"
cargo run --manifest-path crates/rz-experiments/Cargo.toml -- verify "$ARTIFACT_ROOT/runs/e01-input-lock.json" --artifact-root experiments/baselines/fixtures --max-artifact-bytes 1048576
```

독립 패키지의 임시 `Cargo.lock`은 Git에서 제외했다. 최초 빌드는 `cargo check` 등으로
이를 생성하고 이후 `--locked`로 같은 resolution을 사용한다. 인수 실행의 lockfile
digest를 기록한다. workspace의 정식 lockfile은 I01 소유자가 도입한다.

## 형식과 검증 경계

입력은 [작은 JSON fixture](../../experiments/baselines/fixtures/e01-input.json)의 구조를 따른다.
fixture의 CPU·engine·binary identity는 모두 **합성 값**이며 실제 장비 측정·컴파일·대국
기록이 아니다. artifact 텍스트는 검증 대상 바이트일 뿐 실행 가능한 엔진이 아니다.
첫 두 엔진 중 첫 engine ID가 통계의 후보 관점이다. `purpose=fixture`를 실제 비교로
승격하려면 엔진·가중치·옵션 근거와 별도 실제 환경·설정을 제공해야 한다.

JSON은 최대 4 MiB이고 UTF-8이어야 한다. 모든 object의 중복 키와 알 수 없는 필드를
거부한다. source는 query·fragment·userinfo 없는 HTTPS 공개 출처를 사용한다.
옵션 값의 제어 문자는 거부하며, 입력에 남은 placeholder도 잠금 대상이 될 수 없다.
다수의 유효성 오류는 `path`, `code`, `message`로 함께 보고한다. CLI 실패 exit code는 2다.

`schema_version=1`, `lock_version=1`, `canonicalization=rz-e01-json-v1`이다.
canonical digest는 타입으로 읽은 입력의 모든 object key를 재귀적으로 정렬하고,
정수·문자열·boolean·null·array를 compact UTF-8 JSON으로 직렬화한 바이트의 SHA-256이다.
array 순서는 의미를 보존하며, envelope·digest 자기 필드는 해시에 포함하지 않는다.
선택 필드의 생략은 파싱 후 명시적인 null로 정규화된다. 숫자는 이 스키마에서 모두 정수다.
개행·표현 순서가 달라도 같은 의미 입력이면 같은 hash이고, 값이 바뀌면 다른 lock이다.
필드 변경이나 canonical 규칙 변경은 이후 schema/canonicalization 호환성을 검토해야 한다.

내부 search 비교는 같은 evaluator·model/weight·encoding/backend/precision/device/runtime,
weight 비교는 같은 evaluator·encoding/backend/precision/device/search/runtime,
runtime 비교는 같은 evaluator·model/weight·encoding/backend/precision/device/search를 검사한다.
요청·관측 옵션도 동일해야 하며 `allowed_option_change`에 선언한 한 가지 옵션만 예외다.
그 옵션은 두 구성에 존재하고 실제 값이 달라야 한다. 무엇이 실제 search/runtime 변수인지의
의미 확인은 실험 검토와 E02 시작 검증에 남는다. 내부 weight 비교에서 파일 옵션이 달라지면
그 항목도 명시해야 하며, 여러 필수 옵션 변경은 현재 v1의 단일 변수 범위 밖이다.

artifact는 명시 목록만 읽는다. 상대 경로·SHA-256·byte 수·권리·출처를 요구하고,
절대/drive/역슬래시/traversal·비밀 파일 이름·Windows 장치 이름을 거부한다.
디렉터리 handle에 묶인 `cap-std` 경로 탐색을 사용해 중간 디렉터리 교체에도 root 밖을
열지 않는다. symlink와 regular file이 아닌 입력도 거부한다. Unix open은 nonblocking과
최종 symlink 거부를 사용하고 hash 읽기는 선언한 byte 수로 제한한다.
실제 launch 시 새로 파일을 열면 다시 대조해야 한다. E01의 검사만으로 미래 파일 변경이나
실제 옵션 적용·GPU drain·공정성을 증명할 수 없다.

E02는 `decode_json`의 bounded strict reader, `ArtifactRef::validate`의 메타데이터 검사,
`LockedManifest::declared_artifacts`의 identity 목록을 재사용한다. `to_compact_json`은
기존 pretty `to_json`과 동일한 envelope를 compact 형식으로 내보내므로, 다른 잠금 안에
중첩할 때 공백 확장으로 입력 상한을 넘지 않는다. 내용과 digest 의미는 동일하다.

`ArtifactRef::open_verified(root, max_bytes)`는 같은 보안 경계에서 검증한 파일을
offset 0의 std 파일 핸들로 반환한다. Arena는 이를 다시 pathname으로 열지 않고 실행한다.
이름 교체와 동시 inode 내용 변경은 별개이므로 caller는 실행 동안 내용 변경과
artifact root 이름 교체를 막아야 한다. 이 API도 실행·공정성 허가를 뜻하지 않는다.

FEN과 UCI 수순은 최소 형식만 검사한다. 실제 합법성·FEN 의미·완전 상태와 종료 판정은
A의 Position과 E02 독립 참조에 연결한다. FEN-only 입력의 unknown prefix는 그대로 보존한다.
runner 이름·정책 식별 문자열을 실제 지원 증거로 해석하지 않는다.

## 의존성과 통합 인계

신규 자체 코드는 이 crate의 [MIT LICENSE](LICENSE)를 따른다. 외부 엔진·가중치·opening의
권리는 manifest에서 각각 관리한다. 직접 사용하는 의존성은 다음과 같다.

| 의존성 | 용도 | 공개 crate 라이선스 |
|---|---|---|
| serde 1.0.228 | 명시적 스키마 직렬화 | MIT OR Apache-2.0 |
| serde_json 1.0.145 | 제한된 JSON 입출력 | MIT OR Apache-2.0 |
| sha2 0.10.9 | SHA-256 artifact/입력 식별 | MIT OR Apache-2.0 |
| cap-std 4.0.3 | 디렉터리 capability 안에서 파일 조회 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| cap-fs-ext 4.0.3 | 컴포넌트별 symlink 거부와 디렉터리 handle 고정 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| libc 0.2.177 (Unix) | nonblocking/no-follow open flags | MIT OR Apache-2.0 |

총괄에게 제안할 변경은 E crate의 workspace member 등록, 이 의존성·MSRV·lockfile
인수, format/lint/test를 기존 CPU 검사 경로에 연결하는 것이다. 공통 타입 게시 후
contract revision·shared error의 실제 연결을 검토한다. 이 crate는 RequestId·Move·WDL
등 공통 primitive를 복제하지 않는다. 루트 공통 선언은 E 변경에 포함하지 않았다.

인수 조건과 후속 E02/E03 순서는 [E 계획서](../../experiments/baselines/IMPLEMENTATION-PLAN-E.md)를 따른다.
내부 pair 계획·실패 원장·CLI는 [arena README](../rz-arena/README.md)에 기록한다.

## 실제 로컬 검증 — 2026-10-03, Asia/Seoul

기준 `develop`은 `9f0bc598f6b2d8f863fd46af6a4fd73bfef1f0b8`이다.
Linux x86_64·Rust 1.99.0에서 위 test/fmt/clippy 명령을 실행했다.
CLI 9개와 library 22개, 총 31개 테스트가 통과했고 fmt/clippy 오류는 없었다.
`cargo +1.85.0 check --manifest-path crates/rz-experiments/Cargo.toml --locked`도 통과했다.
feature는 기본값이며 자체 엔진·외부 runner는 실행하지 않았다.

별도 artifact root에서 실제 바이너리의 validate→lock→verify를 실행해 합성 파일의
크기·hash까지 확인했다. 독립 Python의 sorted compact UTF-8 JSON SHA-256은
`3763c90ba72e8bf412a84bd187fa2d6e7973994e0a31df3c2c3795075493e567`이며
Rust 잠금과 일치했다. 사용한 임시 Cargo.lock의 SHA-256은
`3f0688d9b1c467bde89a16b69577255c8f50883f8de5f4ff02cf21eabaa01f9b`다.
잠금·CLI 명령 영수증·해당 lockfile은 저장소 밖 `runs/`, `reports/`에 보존한다.
환경 종료 전 회수해야 하며 이를 영구 원격 보존이나 CI 성공으로 해석하지 않는다.

실제 엔진 대국·NN reference·목표 GPU·통계 계산·CI는 미실행이다.
GPU capability는 unknown이며 목표 RTX 4050 6GB 검증은 지원 환경의 별도 인수가 필요하다.
