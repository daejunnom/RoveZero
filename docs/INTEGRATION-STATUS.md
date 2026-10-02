# 총괄 통합 인수 상태

기준일: 2026-10-03, Asia/Seoul. TASK-I01/I02의 source 관찰·수동 연결·실제 검사와
남은 실패를 구분하는 현재 기록이다. 상세 적용은 [CONTRACT-ADOPTION](CONTRACT-ADOPTION.md),
작업 소유는 [IMPLEMENTATION-DIRECTIVES](IMPLEMENTATION-DIRECTIVES.md)를 따른다.
CPU/mock 인수 소스는 `4866a0b67dfd0aac10ffb03a0debd61fa018db3d`, 후속 실제 CPU NN
인수 소스는 `a9a5ae7f5c0bb5b13d8082e2ce662f8e807d8248`다. 각 절의 결과는 지정 소스에서
총괄이 관측한 실제 검사이며 GPU·학습·정식 대국과 구분한다.
문서 후속 수정의 재사용 확인도 실제 실행과 구분한다.
원시 근거는 저장소 밖에 보존하고 논리 경로·digest만 연결한다.

## 기준 조합

공통 계약 [PR #7](https://github.com/daejunnom/RoveZero/pull/7)은 병합됐다.
총괄 통합 시작 develop base는 `3e3cd80533a69ee2118f1c130e72a1cb0745ae8c`,
통합 [PR #9](https://github.com/daejunnom/RoveZero/pull/9)는
`cb9f3d53b5fee1f0f74927d76e9dabd3642b3d9c`로 develop에 병합됐다.
다음은 총괄이 포착·반입한 담당 source pin이다. 반입 뒤 root workspace의 adapter를
수동 수정했다. 담당 standalone 소스와 최종 integration SHA의 실행 결과를 구분한다.

| 담당 | PR | 현재 인수 기준 source head |
|---|---|---|
| A | #3 | `118dc0311a88e143be285703940321dc16261f6a` |
| B | #6 | `9eb805fe7e526d67044b69a4de67c80c08b5aafa` |
| C | #5 | `b295e734f07b01ec65baf8ed6237daaa07ef5aad` |
| D | #4 | `f2a6c605dccfe85f8127b52b76380844a4c9c81c`; 소스 기준 aa2fe1c, 후속 보존 자료는 외부 회수 |
| E | #8 | `91818e3d1309592cff561aa925688446c4a2e4e4`; Rust 접점 수동 정합, 원시 자료·과거 scripts는 외부 회수 |
| F | #2 | `bf38023640b2882e079eafc43e437ea9384b78f9`; 소스 기준 36e619b, 후속 보존 자료는 외부 회수 |

담당 PR #2/#3/#4/#5/#6/#8도 이 조합의 스택된 source 이력을 develop에 포함한 뒤
병합 상태를 확인했다. 원래 head와 실패·연구 근거는 보존하고 history를 재작성하지 않았다.
위 develop merge의 [CI run 37052215045](https://github.com/daejunnom/RoveZero/actions/runs/37052215045)는
Ubuntu·Windows 모두 성공했다.

E `91818e3d1309592cff561aa925688446c4a2e4e4`의 Rust 보완은 `rz-arena/src/fastchess.rs`,
`src/runner.rs`, `tests/fastchess.rs`에 수동 반영했다. 4095 ply 상한, 실행 전
profile 검사, bounded stdout 로그 인자, config.json의 감시·보존·공유 byte 예산과
opening 재검사를 적용했다. 이어 사용자의 자료 정리·develop 반영 지시에 따라
D/E/F 최신 head의 이력을 merge했다. 기존 root 접점·Windows 검사·문서 정합을
보존하고, 원시 자료는 외부 회수 후 최종 소스 tree에서 제외했다. Git에 이미 게시된
원래 source history는 보존하며 재작성하지 않는다. 검토한 요약과 불변 출처만 소스에 둔다.

## TASK-I01: workspace와 단일 계약

- [x] Root Cargo에 10개 member 선언: `rz-contracts`, `rz-position`, `rz-search`,
  `rz-uci`, `rz-encoding`, `rz-eval`, `rz-runtime`, `rz-telemetry`, `rz-experiments`, `rz-arena`.
- [x] Root path 의존에서 `rz-contracts=0.1.0`을 단일 공통 계약으로 지정.
- [x] 선언 확인: toolchain `1.96.0`, workspace rust-version `1.90`; common crate
  MSRV `1.85`는 별도 범위이며 전체 workspace의 1.85 지원을 뜻하지 않는다.
- [x] 실제 root 10개 crate의 `cargo check --workspace --all-features` 성공.
  feature compile 성공은 실제 NN/provider·GPU 장치 실행의 성공과 구분한다.
- [x] A Rules → C `ClassicalProjection`/`ScriptedRuntimeBackend` → D runtime →
  B async UCI binary의 CPU/mock 접점을 연결. provider는 명시적인 `--cpu-mock`이다.
- [x] 실제 소비자의 revision·오류·시간·취소·수명 접점을 수동으로 맞추고
  `4866a0b67dfd0aac10ffb03a0debd61fa018db3d`의 CPU/mock 검사로 인수했다.
- [x] 전체 test·fmt·엄격한 lint·양 OS CPU CI gate가 해당 소스 SHA에서 통과했다.

공통 `ProcessClock`과 ID allocator, 실제 Rules 상태/ordered legal, C의 현재 tick→
취소→ack 순서, D의 admission·물리 완료와 lease, B의 callback 거부·hard timer 출력
권한을 직접 연결했다. 정상 완료의 출력 권한을 runtime 정리가 먼저 취소하지 않는다.
B `RejectedResult`의 원래 오류·context·recovery를 engine에서 소비하고, 진단의
32개 distinct receipt 상한을 넘는 원래 receipt도 보존한다. Failed publication은
worker당 원본 하나를 owner mailbox에 둔다. 정확한 ticket·completion과 실제 실패
진단 소비 때만 확인하며, 채널 종료·queued quit·panic·drain timeout·thread 완료
직전 preemption에서도 회수한다. 동일 join 복제만 제거하고 별도 cleanup 오류는 남긴다.

## TASK-I02: 단계별 증거

| 단계 | 현재 확인한 범위 | 별도 인수와 상태 |
|---|---|---|
| Source 반입·adapter | 위 source pins, 단일 계약 path와 root 10개 crate 전체 target/feature 검사 성공 | 새 head 변경마다 영향 consumer 재검증 |
| CPU/mock binary·trace | A→C→D→B async binary, C bridge 6개, UCI library 33개·실제 binary 7개가 전체 workspace 검사에 포함돼 성공 | 명시적 `--cpu-mock`; NN·GPU 인수는 별도 |
| 실제 CPU NN | 아래 a9a5ae7에서 선정 Maia·고정 LC0 Eigen 참조·ORT CPU와 실제 A→C→D·B UCI 연결 성공 | Linux CPU/FP32·지정 fixture·정상 완료 범위; 실제 Windows NN·목표 GPU는 별도 |
| 목표 GPU | 목표 RTX 4050 6GB와 준비 provider를 개발 호스트와 구분 | 실제 장치·precision·batch·수치·buffer·메모리·D02 종단 계측 pending |
| 실제 F02 학습 | F CPU 숫자 fixture gradient·checkpoint/export/resume 제공; 총괄 Linux F 검사 105개 성공 | 실제 Maia 학습·실제 encoder·검증 data·유한 예산·CONTROL-1→2 인수 pending |
| 정식 paired 대국 | E manifest·pair planner·attempt ledger·원시 WDL/n0..n4와 고정 Fastchess fixture 준비 | Fastchess fixture 전체 실행 not_run; 실제 NN·시계/자원·군집 통계·holdout 및 정식 대국 인수 pending |

이전 E source 리뷰의 test 선언 74개(Unix 조건부 4개)는 당시의 정의 수이며 최신 E
실행 결과로 재사용하지 않는다. F source의 test method 105개와 아래 실제 105개 성공도
정의 수·실행 결과를 구별한다. 정식 실행은 `execution_ready=false`를 유지한다.
F의 fixture 데이터 `training_eligible_records=0`·wire/Rules/encoder `not_run`을
실제 A/C 연결이나 CPU 검사 성공만으로 승격하지 않는다.

## 후속 실제 CPU NN 인수

CPU/mock과 연구자료 인수 후 [PR #10](https://github.com/daejunnom/RoveZero/pull/10)에서
선정 Maia1 가중치를 실제 CPU 경로에 연결했다. 인수 소스는
`a9a5ae7f5c0bb5b13d8082e2ce662f8e807d8248`이며 아래 과거 CPU/mock 인수와 구분한다.

첫 소스 `48eb35063f1a41648c9d77fb4ea5821ead5e12f6`은 C의 physical output validator
오류와 물리 완료 미확정 worker panic의 bounded per-job 원인을 보존한다. 기존 common
output API를 유지하며 quarantine를 Ready로 처리하거나 입력·session pin을 해제하지
않는다. 해당 변경의 Linux/Rust 1.96.0 `contracts_adapter` 6개·`physical_worker` 4개가
성공했다. 이 결과는 native runtime·UCI 전체 연결이나 실제 NN 실행 증거가 아니다.

선정 원본을 외부 CPU 작업 root에 준비해 압축·protobuf 해시를 다시 대조했고, ORT
1.22.0 CPU library의 실제 파일 identity를 기록했다. 외부 LC0
`fd71a2d921b689c5f479d3227c3806c8e272d9c5`와 common submodule
`b326b154221a6eb91977bdccf11d6f89e8547875`를 고정하고 converter·Eigen Python
참조를 별도로 빌드했다. 제품에 LC0를 링크·복제하거나 UCI wrapper를 neural API로
대체하지 않았다. 기존 원본 protobuf의 Eigen 출력과 RoveZero를 실제 대조했다.

Ubuntu/Python 3.12.3·Rust 1.96.0, ONNX Runtime 1.22.0·CPU·FP32로 아래를 실행했다.
각 native 검사 child에 2 GiB address-space 상한과 60초 외부 timeout을 두고 CPU
thread 1·저장소 밖 output root를 사용했다. 이 상한은 실제 peak RSS/VRAM 측정치가
아니다. LC0 도구 빌드의 aggregate compiler peak RSS는 미검증이다.

| 검사 | 실제 결과와 범위 |
|---|---|
| `maia_check` | 12개 독립 참조 case의 raw 1858 policy logit·합법 policy·WDL, batch 1/2/4/8/16·오류 입력·C worker 계약 검사 성공. 직접 frame fixture이며 A/D 경로와 구분 |
| `rules_maia_check` | 실제 A FEN·trace·immutable export·ordered legal → C ONNX → D 완료 12개 성공. history-fill No 5개/RepeatOldest 7개, dense 입력 오차 0, 합법 policy 최대 절대 오차 `1.1920928955078125e-6`, WDL `4.76837158203125e-7`; 기준은 각각 `1e-4` |
| Rules 경로 drain | 두 history profile의 C 진단·boundary·poison 없음. D reserved host/device/pinned/request 모두 0, executions/logical requests 0·physical Drained |
| 실제 CPU UCI | 같은 process에서 시작·흑 차례·양색 캐슬링/승격 가능 상태 6개에 `go nodes 4 movetime 2000`, 각 합법 bestmove·중복 없음·마감/실패 fallback 없음. `ucinewgame`·stop의 현재 root 합법 착수·quit exit 0·owned group/reader 정리 성공 |
| Native UCI 완료 근거 | D가 수락한 `CpuOnnx`·FP32·fullsteps 1·Computed 출력 23개. 물리 호출 수·방문 수와 구분. 현재 UCI는 nodes 관측을 노출하지 않으며 stop 시 물리 in-flight는 unknown |
| 잘못된 manifest pin | exit 2·UCI 출력 0·Asset/IdentityMismatch; CPU/mock으로 대체하지 않음 |
| 현재 소스 검사 | workspace 560 성공·0 실패·12 ignored, 기본 feature CLI 4 성공, Rules release 독립 대조 42 성공·0 ignored, F fixture 105 성공, fmt·Clippy 성공 |
| [CI run 37059463876](https://github.com/daejunnom/RoveZero/actions/runs/37059463876) | a9a5ae7 head 직접 checkout, Ubuntu·Windows 모두 필수 단계 성공. 실제 모델 asset을 사용한 NN 실행은 위 로컬 Linux 검사이며 CI에 포함되지 않음 |

실제 UCI의 여섯 finite 결과는 2초 마감보다 이른 1.5초 인수 guard를 통과했다.
이 작은 smoke의 응답 시간은 benchmark·처리량·대국 강도 근거가 아니다. Rules 예제의
model/encoding handle은 명시한 로컬 fixture 소유이며 production registry 발급의
검증이라고 보고하지 않는다. UCI bootstrap의 실제 owner 조합 검사는 별도 실행이다.

| 외부 입력/도구 | SHA256 |
|---|---|
| 원본 `maia-1900.pb.gz` | `e2f565f42d7cd9f122557e6dc4eb84e5bbaedceda1d404dc485d3611c7c97a12` |
| 원본 압축 해제 protobuf | `e8fe5a7f35594d4190c440a2716fda57ba9d071b26175d509476be7a4fda052a` |
| ONNX export | `25a0a378d4dfea3c4df71af825006713e221be93e47d5b2e5440e5d882138f77` |
| Export manifest | `63e4f9c28f2ce798c0f241583bbb338bffb89ad058ccf7117d3813eff441a0c9` |
| ORT CPU 1.22.0 library | `7520bc1b4f649ee8fa956442a66ded303e8fd3b465b0159b7e893bdc87df4e38` |
| LC0 Eigen 12-case reference | `b2f9130b62c2333f189084ce305ace82166b6b6464fa83357f33affed92f52bb` |
| 실행한 UCI 검사 driver | `e40a803c1520ce794e6c25338da5d2c8354768f2c32cc93dd6d180be2be55668` |

외부 GPL 가중치·변환본·LC0 도구를 MIT 엔진 소스와 분리했다. raw 결과·로그·도구 준비
영수증은 `reports/coordinator-integration/native-cpu/`에 보존하며 결과 JSON digest는
`legacy-maia-a9a5ae7.json`: `88aee5fcc47e8252017c70f92e0469d57008ce2609f04d81f5bfc8a25930a09e`,
`rules-maia-a9a5ae7.json`: `ec85f4c980ed449ff7e1785d942fd737c4e35290e98e22421a07a84207aa07a2`,
`native-uci-a9a5ae7-final.json`: `7ebdc3301b83ae4cb7bcdbe83158dade1985c3c07b97506dea37d1f1c3b4ffe7`다.
실행한 driver 사본을 별도로 보존했으며 그 후 double-failure probe의 원인 보존 수정은
미실행 경로 개선으로 구분한다. 기존 성공 결과를 수정한 driver의 실행으로 보고하지 않는다.

과거 실패도 보존한다. uppercase hash fixture 두 개와 예제 Clippy를 고쳤다.
첫 UCI driver는 제품에 없는 `info nodes`를 요구해 실패했으며 제품 코드를 바꾸지
않고 미관측을 명시했다. 독립 리뷰에서 마감 fallback의 허위 통과 가능성과 leader
종료 뒤 child group 누락을 발견해 검사 경계를 수정했다. 성공한 마지막 driver는
그 두 수정 뒤 실행했다. 최종 C/D와 B source 읽기 리뷰는 추가 차단 결함을 찾지 못했으며
actual file digest를 외부 영수증에 보존한다. GitHub approval 리뷰와는 구분한다.

Linux 실행 자료 30개·1,025,003 bytes를 Windows 보존 root에도 byte 그대로 회수해
각 SHA256를 대조했다. `captured-execution/capture-receipt.json`의 digest는
`67fd719137bdb197b6cd91885e0e40e4ef16ddc2f71c907ff5d367ffa9030c5a`,
`native-acceptance-a9a5ae7.json`은
`4e4ae73ecc4a4b03f9ef23eb1bebc3befcebd5b0b84b4a8d0db7c2936f0614cb`다.
원시 자료는 소스 tree에 포함하지 않으며 외부 작업 root의 도구·모델은 그대로 보존했다.

첫 연결은 명시적 CPU provider·FP32·session 1개·thread 1개·batch 1을 사용한다.
root 교체 중 이전 물리 worker가 남으면 기존 WorkerLimit 진단과 합법 fallback을
유지한다. 루트별 유한 탐색 예산과 process 공통 clock/ID를 사용하며, 성공한 요청마다
누적 identity 목록을 늘리거나 일정 횟수 후 실제 평가를 소진시키는 설계는 채택하지 않는다.
native 원인의 소유자·진단 전달·물리 drain도 별도로 검사한다. 실제 GPU·학습·정식
대국 인수는 이 CPU gate 이후에도 별도 상태를 유지한다.

## 실제 검사 영수증

아래는 모두 integration `4866a0b67dfd0aac10ffb03a0debd61fa018db3d`의 실제 실행이다.
로컬은 Ubuntu/Rust 1.96.0·Python 3.12.3, `CARGO_BUILD_JOBS=2`, 저장소 밖의 작업 전용
target/report root를 사용했다. provider는 CPU/mock이며 가중치 파일을 사용하지 않았다.
서로 다른 gate의 검사 개수를 합쳐 하나의 강도·GPU 성공 수치로 표시하지 않는다.

| 검사/관측 | 실제 결과와 범위 |
|---|---|
| `cargo fmt --all -- --check` | 로컬·Ubuntu CI·Windows CI 성공 |
| `cargo test --workspace --all-targets --all-features --locked` | 로컬/Ubuntu CI: 535 성공·0 실패·12 ignored, 55 target 결과. Windows CI: 515 성공·0 실패·2 ignored |
| `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | 로컬·Ubuntu CI·Windows CI 성공 |
| `python -m unittest discover -s experiments/model-research/tests -q` | 로컬·양 OS CI 각각 105 성공; `PYTHONPATH=experiments/model-research/src` |
| `cargo test --release -p rz-position --all-features --locked --test perft -- --include-ignored` | 로컬 3 성공 |
| `cargo test --release -p rz-position --all-features --locked -- --include-ignored` | CI의 pinned python-chess 1.999/chess 1.11.2 독립 대조 포함, 양 OS 각각 42 성공·0 ignored. 독립 oracle은 로컬 미실행 |
| [CI run 37048414963](https://github.com/daejunnom/RoveZero/actions/runs/37048414963) | PR head를 직접 checkout; 해당 SHA의 양 OS 모든 필수 단계 성공 |
| CPU NN·목표 GPU·실제 학습·정식 arena | 4866 당시 root 실제 실행 인수 pending; 후속 Linux CPU NN은 위 절에서 인수. 나머지 gate는 지원 환경·입력·유한 예산 별도 잠금 |

Ubuntu ignored는 E native child helper 10개와 A 확장 perft·독립 oracle 2개다.
E helper는 supervisor 검사에서 별도 child로 실행하는 진입점이며 전체 workspace에
`--include-ignored`를 붙여 직접 실행하지 않는다. A의 두 검사는 CI release 단계에서
실제 실행했다. 플랫폼 조건부 검사 때문에 두 OS의 workspace 개수가 다르다.

Publication 수정의 신규 회귀 11개는 실제 dispatch, 닫힌 receiver, queued quit,
소비 확인 후 다음 root, 진단 poison, shutdown panic, 미완료 drain의 원인 회수와
closure 반환→thread 완료 사이 preemption을 검사한다. 원인 회수는 물리 완료의
주장이 아니다. 독립 읽기 리뷰는 추가 차단 결함을 찾지 못했으며, 검토한 engine 파일
SHA256 `5c517a3ca4f7b36429a82d4f11996bbee0ca4618728ac92d0d9a15217930d86e`가
인수 커밋의 파일과 일치했다. GitHub approval 리뷰는 아직 없으며 이 독립 리뷰와 구분한다.

이전 실패 기록은 유지한다. [run 37040889696](https://github.com/daejunnom/RoveZero/actions/runs/37040889696)의
B deadline/output·Windows E rename 실패, [run 37043890313](https://github.com/daejunnom/RoveZero/actions/runs/37043890313)의
Windows clock·E lint 실패, [run 37045100732](https://github.com/daejunnom/RoveZero/actions/runs/37045100732)의
Windows pin lint 실패를 각각 수정했다. 원래 guard·실패 원인을 제거해 통과시키지 않았다.
`23d3567`의 양 OS 성공 뒤 독립 리뷰에서 찾은 publication 원인 손실도 후속 커밋과
11개 회귀로 수정했다. 이전 SHA의 일부 성공 개수를 최신 성공 개수에 합하지 않는다.

검증 입력의 Git object ID는 외부 파일 SHA256과 구분한다. 문서 후속 수정은 이 입력의
검사 당시 입력을 식별한다. 4866→PR #9의 연구자료 인수 뒤 문서 후속 구간은 Markdown에 한정돼 있으며,
실행 영향 src/tests/examples·Python src/tests/fixtures·manifest/lock/toolchain/CI는
동일하다. 이 소스 검사의 재사용 확인과 최종 PR head에서 새로 실행한 CI는 구분한다.

| 입력 | 인수 소스의 Git object ID |
|---|---|
| `crates/` tree | `0647cefca478d9eab22e61b62e6bf12322277376` |
| `experiments/model-research/` tree | `a4cf397ca7820f6dbe12c93f828fb772a8076821` |
| `Cargo.toml` blob | `a786dd7c3f3bb03633d90bd070e87fad6157e6a6` |
| `Cargo.lock` blob | `40ae7ed3acc260b31746b0b78c5d486b0e69e23f` |
| `rust-toolchain.toml` blob | `c2294d8fb5bac0db761c2d5016932a38ab5060bc` |
| CPU workflow blob | `748ed8e119ec3101e1f82370edc3bc950dde5aa8` |

보존한 실행 로그는 저장소 밖 `reports/coordinator-integration/`에 있다.

| 논리 파일명 | SHA256 |
|---|---|
| `workspace-4866a0b.log` | `4d0e408ebdb32fdd5bf5e3a218e8bcbe7589abb21a060d102476433cd29974bb` |
| `clippy-4866a0b.log` | `b5cd628ef83643c4d1d3ef02e10d689554460495f0b7c843866175423229357d` |
| `python-4866a0b.log` | `6df049697dc52e2743bdfef7341f4acbf02ee51056bd368f30bde220df7f1e77` |
| `perft-4866a0b.log` | `d05b1ffc49582bc299746a1a7517c54f6d78573c1a0ec910dd7a87481f8c798a` |
| `ci-4866a0b-37048414963.log` | `f4d9c1fc8701d17c90f22b47bbccf81a923b994703fe966f2b974e1cbe69f479` |

## 증거 보존과 회수 상태

사용자는 앞서 회수를 보류한 뒤 D~F 자료가 PR에 게시됐음을 알리고 정리·develop 반영을
지시했다. 이에 **검증된 공개 보존 사본을 회수했다.** 마스킹 전 클라우드 원본,
처음부터 저장하지 않은 로그·제외 binary·부재 wheel까지 회수한 것은 아니다.

모든 파일의 mode·prefix·크기를 먼저 확인하고, 저장소 밖 사본의 길이와 Git blob
identity를 지정 source와 대조했다. Windows의 자동 줄바꿈 변환을 거친 사본은
인수하지 않았고, 최종 exact-blobs 사본은 719개 파일 모두 source byte와 일치했다.
별도 과거 E script 두 개도 byte를 대조해 보존했다. 재현 script를 실행하거나 새로운
과거 실험을 만들어 누락을 채우지 않았다.

| 담당 | 공개 사본 수 / bytes | 회수의 논리 slot | archive SHA256 |
|---|---|---|---|
| D | 26 / 333,588 | `exact-blobs/D-f2a6c60` | `3e037102404a20e6863be201ea942e9fd372189c61c1a63535dcee5a0ac42257` |
| E | 454 / 1,786,455 | `exact-blobs/E-91818e3` | `2b364c28ec5241402698651063a242c0e1a4b924c57f997e70e748bcafbf7bdf` |
| F | 239 / 917,150 | `exact-blobs/F-bf38023` | `bc60a9dbcb44ba8a7c1256de7c196f12b0a622bd79838a973126f6e4869b17bb` |

상위 논리 root는 저장소 밖 `reports/coordinator-integration/recovered-pr-evidence/`다.
각 slot의 `.tar`와 `.receipt.json`은 source SHA·prefix·file count·mode·Git blob·
file SHA256를 보존한다. receipt SHA256는 D
`9614c9f75ca657a7cd70183ce1c896255b2b83a696b7944c7a88be786e77f693`, E
`24d43849597b49eeeb1bd8003581c85b170f43cf930b7709793e4f99888502b2`, F
`ffc74aa0cad13d8d84e38804e155f9ca54125a779e16c5189e6a3548a45290cb`다.
E 과거 scripts의 별도 receipt SHA256는
`0a523b4c08a587ab55083dde1ca30ca7a354403004654e2347d9d48edecf78a7`이다.
후속 인수와 별도 정리 지시까지 보존한다. 원래 클라우드 scratch와 공개 Git 이력은 지우지 않았다.

독립 자료 감사에서 확인한 범위와 한계는 다음과 같다.

- D: 원본 TSV 8개·184,296 bytes·4,082 data rows와 results의 760개 metric이 맞았다.
  재수집 69 tests는 aa2fe1c의 runtime 51 + telemetry 18이다. long-loss/zero의
  관측 drop·오류와 불완전 profile, 초기 4개 source unknown, 정확한 과거 SHA/argv/UTC·
  요청별 입력·event stream 누락을 유지한다. CPU 합성 stage·mock 계측이며 GPU/D03 성과가 아니다.
- E: 네 inventory의 자료 448개와 11 ProcessReceipt가 참조한 artifact 46개가 맞았다.
  과거 source 9e49b667의 Linux standalone 140 tests와 synthetic Event 회계를 분리한다.
  실제 프로세스의 7수 script 1승·1패, cutoff Incomplete, signal 15 취소와 cleanup
  Unverified도 보존한다. 시작 lock snapshot·최초 build raw 로그·제외 binary가 없으며,
  마스킹 전 원본 회수나 root 엔진·NN의 대국 성과가 아니다.
- F: manifest 237개·SHA256SUMS 238개, embedded digest 83개와 F02 receipt 11개가 맞았다.
  source 미기록 historical 22개(실패 포함), source 36e619b의 새 recheck 13개,
  재구성 audit·파생 TSV·경로 마스킹본·부재 wheel을 구분한다. 연속/resume의 model·
  optimizer·sampler·history 정합은 numeric float64 fixture이며 실제 Maia 학습이 아니다.

각 공개 bundle의 과거 remote receipt는 그때 대상 commit에만 적용한다. 이번 최신
head의 독립 byte 감사와 총괄 외부 회수 receipt, 현재 root 실행은 서로 다른 근거다.
불변 출처는 [D](https://github.com/daejunnom/RoveZero/tree/f2a6c605dccfe85f8127b52b76380844a4c9c81c/benches/runtime/evidence/aa2fe1c-d02),
[E](https://github.com/daejunnom/RoveZero/tree/91818e3d1309592cff561aa925688446c4a2e4e4/experiments/baselines/evidence),
[F](https://github.com/daejunnom/RoveZero/tree/bf38023640b2882e079eafc43e437ea9384b78f9/experiments/model-research/verification/2026-10-03-f-evidence-v1)다.

E의 과거 `check-e.py`/`run-e-fixture.py`는 단일 root lock·실제 binary SHA 연결,
출력 상한·readiness/timeout 자식 정리가 맞지 않아 실행 도구로 인수하지 않았다.
원래 소스는 불변 commit과 외부 사본에 보존하며 실제 root 도구는 별도 정합·검사를 요구한다.

증거는 source 관찰 → 대상 SHA의 실제 실행 → 외부 자료 보존·회수 → digest/입력·환경
대조 → 해당 gate 인수로 구분한다. 위 새 로컬 CPU/mock·F fixture 로그는 총괄의 새
실행 근거이며 과거 클라우드 자료의 대체 회수가 아니다. 새 검사와 공개 자료 감사,
아직 회수하지 않은 마스킹 전/누락 원본을 혼동하지 않는다. 이후 회수에서도
원시 로그·TSV·checkpoint·실행 영수증을 Git에 반입하지 않고 외부 보존 참조를 연결한다.

각 실제 인수에서 총괄은 integration SHA, 담당 full source heads, 계약 revision, 명령·OS·
toolchain·backend/feature·fixture/weights/config digest·자원 한도, 실제 결과·실패·
skip/미실행, 외부 산출물 참조·digest와 다음 검증자를 기록한다. CI 요청·관측 완료·
재사용 확인도 분리한다. CPU/mock과 후속 Linux 실제 CPU NN 소스·gate는 위에서
확정했고 목표 GPU·GPU 종단 계측·학습·정식 대국은 각각 pending이다. 고정 Fastchess fixture
전체 실행도 `not_run`이며 정상 PGN·exit 0만으로 raw UCI 로그 완전성을 인수하지 않는다.
후속 총괄은 사용자 재개 때 원격 head와 이 조합을 대조해 계약을 수동 정합한다.
