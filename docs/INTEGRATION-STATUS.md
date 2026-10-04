# 총괄 통합 인수 상태

기준일: 2026-10-04, Asia/Seoul. TASK-I01/I02의 source 관찰·수동 연결·실제 검사와
남은 실패를 구분하는 현재 기록이다. 상세 적용은 [CONTRACT-ADOPTION](CONTRACT-ADOPTION.md),
작업 소유는 [IMPLEMENTATION-DIRECTIVES](IMPLEMENTATION-DIRECTIVES.md)를 따른다.
CPU/mock 인수 소스는 `4866a0b67dfd0aac10ffb03a0debd61fa018db3d`, 후속 실제 CPU NN
인수 소스는 `a9a5ae7f5c0bb5b13d8082e2ce662f8e807d8248`다. 각 절의 결과는 지정 소스에서
총괄이 관측한 실제 검사이며 GPU·학습·정식 대국과 구분한다.
문서 후속 수정의 재사용 확인도 실제 실행과 구분한다.
원시 근거는 저장소 밖에 보존하고 논리 경로·digest만 연결한다.
후속 GPU A/B 이전의 의미 보존 최적화 제품 소스는
`1cdd707922dcaaf234b4cd162210ce9f2fa512bb`다. CPU workspace 675개·strict clippy와
양 OS CI, 같은 상태 witness와 CPU 교차 측정을 확인했다.
[사전 최적화 기록](research/PRE-RUNPOD-OPTIMIZATION.md)을 따르며 현재 소스의 실제 NN/GPU
실행으로 확대하지 않는다. Community 후보·network volume 미사용·초기 총예산 약 US$200과
Oracle 보존 경로는 [외부 계획](research/RUNPOD-BENCHMARK-PLAN.md)에 기록한다.
실제 CPU 신경망의 Fastchess pair 연결 인수 소스는
`0f0d70dddb170e7d46729eb19da02e810f0dee61`이며 후속 절에 별도로 기록한다.
첫 목표 CUDA 수치·provider 인수 소스는 `bde687c7f269af3c7cd501201edde14c5c8ef642`다.
이 후속 검사는 C worker·fixture Rules 범위이며 실제 A/D GPU 종단 인수와 구분한다.

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

- [x] Root Cargo의 초기 10개 member 선언: `rz-contracts`, `rz-position`, `rz-search`,
  `rz-uci`, `rz-encoding`, `rz-eval`, `rz-runtime`, `rz-telemetry`, `rz-experiments`, `rz-arena`.
  후속 CUDA bootstrap에서 `rz-native-loader`를 추가하여 현재 member는 11개다.
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
| 목표 GPU | 아래 CUDA provider 수치·12개 실제 A 상태→D·B UCI와 e45dd10의 실제 CUDA integration pair를 각각 인수 | 첫 실패 보존; GPU 수명의 모든 조합·process peak VRAM·D02 종단 계측 pending |
| 실제 F02 학습 | F CPU 숫자 fixture gradient·checkpoint/export/resume 제공; 총괄 Linux F 검사 105개 성공 | 실제 Maia 학습·실제 encoder·검증 data·유한 예산·CONTROL-1→2 인수 pending |
| E 합성 fixture | 고정 Fastchess의 실제 paired 실행·A/독립 PGN 대조·artifact 사전 거부·live 취소를 아래에서 인수 | 합성 script·Linux CPU만 해당; NN·GPU·정식 시계/자원과 구분 |
| 정식 paired 대국 | E manifest·pair planner·attempt ledger·원시 WDL/n0..n4와 아래 CPU/CUDA NN integration pair 제공 | 정식 시계/자원·군집 통계·holdout 및 강도 인수 pending |

이전 E source 리뷰의 test 선언 74개(Unix 조건부 4개)는 당시의 정의 수이며 최신 E
실행 결과로 재사용하지 않는다. F source의 test method 105개와 아래 실제 105개 성공도
정의 수·실행 결과를 구별한다. 정식 실행은 `execution_ready=false`를 유지한다.
F의 fixture 데이터 `training_eligible_records=0`·wire/Rules/encoder `not_run`을
실제 A/C 연결이나 CPU 검사 성공만으로 승격하지 않는다.

## 후속 E 고정 Fastchess fixture 인수

[PR #11](https://github.com/daejunnom/RoveZero/pull/11)의 실행 소스는
`3de9603c94da68bf3fa0f47797a1f37559e0ec95`, 후속 검사 소스는
`d15485017d3affdccf0df274df63fd988fa6f3e3`다. 후자는 실제 binary 검사 helper의
원래 실패·독립 cleanup 실패 보존만 바꿨다. diff의 변경 경로가
`crates/rz-arena/tests/fastchess.rs` 하나이며 제품 실행 소스·Cargo·toolchain·CI는
일치함을 확인했다. 이전 release binary의 source SHA를 새 SHA로 바꾸지 않았다.

선택적 UCI `FixtureMode=normal/crash/illegal/timeout`을 실제 합성 binary에 연결하고
CLI 초기값·새 게임 이후 설정 보존·잘못된 값 거부를 검사했다. 기존 builder의
Fixture·CPU·무가중치 제한은 유지하고 제품 엔진에는 이 옵션을 허용하지 않는다.
계약 revision은 0.1이다. Linux/Rust 1.96.0에서 release binary 세 개를 실제 빌드해
digest·byte·명령·root lock·feature를 고정했고, d154850의 E 검사는 **148 성공·0 실패·
10 ignored**였다. ignored는 supervisor 검사가 별도로 띄우는 helper 진입점이다.
[CI run 37064762859](https://github.com/daejunnom/RoveZero/actions/runs/37064762859)는
d154850을 직접 checkout했고 Ubuntu·Windows의 필수 단계가 모두 성공했다.
이 CI는 외부 Fastchess 전체 실행을 포함하지 않는다.

외부 Fastchess 소스 `f618e34540f94f4719ad3817950618dabe441318`, tree
`7af51c972096164b267d617ea4c32a856a17b586`를 clean 상태로 고정해
GCC 13.3.0·C++17·`-march=x86-64`·`ZLIB=false`·jobs 2의 실제 `make all`로 빌드했다.
`build=release`의 정적 링크 설정은 사용하지 않았다. 실제 binary는 2,429,320 bytes,
SHA256 `ca85b6f3cbaab62352d7c98fb684f427a15c8a03d67825f0f2909eae725d9cfe`,
실제 version stdout은 `fastchess alpha 1.8.2 20260726-f618e34`와 newline이다.
본체 MIT와 포함된 외부 공지를 분리 보존했고 POSIX `argv_split.hpp`의 upstream
permission 근거는 unresolved다. 내부 fixture 실행이며 외부 도구의 배포 권리 확정이 아니다.

총괄의 새 외부 driver는 과거 standalone scripts를 재사용하지 않았다. 고정 빌드
영수증·binary를 검증해 별도 snapshot을 만들고 새 manifest→lock→artifact verify→
plan→실제 fixture→trusted tip ledger audit를 실행했다. 총 23개 CLI 명령이 원래 실패와
cleanup을 독립 보존한다. Linux 6.6.87.2 WSL2·Python 3.12.3·default SIGCHLD·subreaper
비활성으로 실행했으며, child에 논리 CPU 0 affinity·프로세스별 address-space 2 GiB를
상속했다. kernel child quota·합산 RAM 한도·peak RSS는 인수하지 않는다.

| 실제 gate | 결과 |
|---|---|
| Normal | 같은 `startpos + e2e4 e7e5`에서 색을 교환한 두 판. 전체 7 ply mate를 A Rules와 독립 `chess 1.11.2`로 대조. runner exit 0·Gone, 완료 pair 1·W1/D0/L1·`n=[0,0,1,0,0]` |
| Cutoff | 전체 4 ply에서 두 판 종료. A/독립 oracle이 진행 중 상태를 확인. PGN의 adjudication draw를 점수에 넣지 않고 Incomplete 2·제외 pair 1·WDL/n 모두 0 |
| Invalid artifact | runner SHA를 다르게 잠근 새 입력. artifact verifier와 실제 fixture 명령이 exit 2로 거부. attempt 디렉터리·runner·engine spawn 없음 |
| Live cancel | 실제 CLI와 별도 runner group·fixture 두 개의 계보/PID/start-time/executable/affinity를 관측한 뒤 CLI에 SIGTERM. E `Cancelled`·signal 15·Gone·오류 없음, CLI exit 2. Incomplete 2·제외 pair 1·WDL/n 모두 0 |

live cancel의 outer cleanup도 Gone·잔여 pin 없음이며 outer TERM/KILL·subreaper adoption은
사용하지 않았다. 이것은 실제 프로세스 실행 중 취소이며, `go` 수신·신경망 physical
in-flight·runtime/GPU drain을 관측했다는 뜻은 아니다. 취소의 PGN 감사는 수행하지 않는다.
기존 receipt의 generic audit 오류 문구는 `Exited + exit 0 + Gone` 완료 조건의 실패를
나타내며, 실제 cleanup은 별도 `process.group_cleanup=gone` 필드로 확인한다.
Fastchess의 자체 정상 quit/reap 경로와 강제 종료 뒤 OS에서 관측한 group 부재도 구분한다.

| 보존 근거 | SHA256 |
|---|---|
| 실제 RoveZero build receipt | `38af224d57c08ffd10ee843d48532960d02b92ea09077d1e75d0dd8066121011` |
| Fastchess preparation receipt | `a26c5007bcedd6f6f1a4c189e2e97bdd910a1e4cbded2fdde092df3c5ae7496d` |
| 실행한 driver | `80b7e74429f3c9cbde1bf314beb4cf965e5737fe02c1da871de8e333e6e9c221` |
| 4 gate acceptance | `6691dc0620cf977c8c8ae17af3677fe2040aac1bb3496a2e33210b8922761019` |
| Windows 회수 receipt | `659426e2c3426c297b6c1d52a656f13ccbe81a311909a48d20ca590c3976592d` |

논리 보존 root는 저장소 밖 `reports/coordinator-integration/arena-fixture/`다.
`captured-execution-baseline-01/`의 171개 파일·589,423 bytes가 Linux의 실행 자료와
SHA256·byte 모두 일치했다. compiler/source checkout·executable snapshot·build object는
별도 외부 root에 있으며 이 회수 수에 포함하지 않는다. fixture의 PGN·raw UCI stdout,
manifest/lock/plan·원장·23개 명령·프로세스 receipt를 보존했다. trace 파일의 존재를
모든 UCI 로그·옵션 적용 또는 provider attestation의 완전성으로 승격하지 않는다.

네 gate는 `execution_ready=false`인 합성 script 인수다. 실제 NN pair에는 별도 typed
launch와 original weights/ONNX/export manifest/ORT pin, 적용 provider와 종료/drain
영수증, unresolved child와 입력/output pins를 함께 보유하는 owner가 필요하다.
현재 native UCI는 고정 CPU 시작 인자를 사용하고 UCI options를 제공하지 않으므로
fixture의 Threads/Ponder 옵션으로 이 계약을 대신하지 않는다. NN pair·목표 GPU·학습·
정식 시계/자원/통계·LC0 대비 강도는 각각 pending이다.

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
전체 실행은 위 새 4 gate에서 인수했으며 정상 PGN·exit 0만으로 raw UCI 로그 완전성을 인수하지 않는다.
후속 총괄은 사용자 재개 때 원격 head와 이 조합을 대조해 계약을 수동 정합한다.

## 후속 실제 CPU 신경망 pair 인수

[PR #12](https://github.com/daejunnom/RoveZero/pull/12)은 기존 실제 CPU NN 엔진과
E runner를 총괄이 수동 연결한 후속 인수다. 실행 소스는
`0f0d70dddb170e7d46729eb19da02e810f0dee61`이며 공통 계약 revision 0.1은 유지한다.
E의 `IntegrationPairSpecV1` wire version 1은 강도 실험과 별도 선언이다. 같은 모델·탐색을
두 역할에 쓰는 첫 연결 검사에서 허위 InternalSearch/Runtime/Weights 변경을 선언하지 않는다.

총괄은 B startup/final의 실제 선언, C 모델·backend·encoding identity, D completed/context,
E private snapshot·supervisor·PGN을 직접 대조했다. 지원하는 인자는 `--onnx-cpu`,
`--attestation`과 명시적인 `--key=value`이며 Fastchess의 하나의 `args=` 값으로 전달한다.
새 프로세스 네 개의 ID와 영수증을 구별하고 `restart=on`으로 게임 사이 상태를 초기화한다.

| 실제 로컬 검사 | 결과 / 적용 소스 |
|---|---|
| Workspace `cargo test --workspace --all-targets --all-features --locked` | 606 passed, 0 failed, 16 ignored; `0f0d70d` |
| Workspace clippy `--all-targets --all-features --locked -- -D warnings` | 성공; `0f0d70d` |
| Release `rz-uci/onnx-cpu`, `rz-arena`, `rz-experiments` build | 성공; `0f0d70d`, Rust 1.96.0, Linux |
| 실제 `native-lock` / `native-pair` | 각각 exit 0; 잠금 선언 검사와 실제 실행을 구분 |
| 실제 CPU NN pair | 두 판 모두 6 ply; e2e4/e7e5 2 ply opening과 이후 실제 착수; 흑백 배정 교환 |
| Native provider / 종료 | 네 프로세스 각각 D completed 258, first/last fresh computed 일치, confirmed drain |
| Runner 감독 | exit 0, process group Gone, pending 없음, 오류 없음; 감독 3.0078초, 외부 실행 3.2068초 |
| 동일 실행 소스 CI | [run 37073237805](https://github.com/daejunnom/RoveZero/actions/runs/37073237805), Ubuntu·Windows 모두 성공 |

16 ignored는 helper 진입점과 별도 심화 규칙 검사이며 로컬 기본 suite의 성공 수에 넣지 않는다.
위 CI의 release 규칙 검사 `--include-ignored`와 Python 학습 fixture는 별도 실제 step으로
관측했다. Windows CPU CI 성공이 Linux 전용 actual launcher 실행이나 GPU 인수를 대신하지 않는다.

선정한 Maia1 v1.0 원본·변환 ONNX·manifest·ORT 1.22.0은 이전 실제 CPU 검사와 같은
검증된 파일을 사용했다. `HistoryFill=No`, FP32, batch/intra-thread/search-worker/full-step=1,
CPU provider, fresh-only를 잠갔다. 원본과 변환본의 외부 권리와 converter lineage를 유지한다.
입력 canonical SHA는 `6540ae15e7d609fe672de39c086d841015cf85ef6be264ebf3bbb9709934b53e`다.

| 실행 파일 | bytes | SHA256 |
|---|---|---|
| `rz-uci` | 2,509,344 | `9621046985fd30d46c0e64a198c7174637fd64f4e67252fa7547f98708f2a4ae` |
| `rz-arena` | 3,202,992 | `5a3b502ee281bc7f192a68f61687f9935f89c1b0f3246a2ce5652ae95648628d` |
| 고정 Fastchess | 2,429,320 | `ca85b6f3cbaab62352d7c98fb684f427a15c8a03d67825f0f2909eae725d9cfe` |

release build receipt SHA는 `7dff26dc56358b8b57b5a658183267226d3346fd6926aa846a04de8ff29f3794`,
실제 pair receipt SHA는 `403afca816b953efbe4692ca93187837ffac1e28e8d471b6410f8ce75922b139`,
PGN SHA는 `cfaa0ebfb44e2479d9b6a0837d14c3b232c16a6dd8aee306d3e07355693d7367`다.
원시 PGN·config·stdout/stderr·provider 기록·선행 실패 로그와 실행 영수증은 저장소 밖
`reports/coordinator-integration/native-pair/`에 보존한다. 알려진 자료 51개·206,260 bytes를
Windows의 `captured-0f0d70d/`로 회수하고 byte 길이·SHA256를 각각 재대조했다.
회수 receipt SHA는 `835454b545cf5eaf49cc30c134deb18ed656f4138e25d5ab7f23d30663552bc1`이다.
가중치·binary·runtime library는 이 로그 회수에 포함하지 않으며 기존 외부 원본과
Linux의 독점 snapshot을 보존한다. 원시 로그에 담긴 개인 경로는 소스 문서에 반입하지 않는다.

자원 설정은 pair 전체 runtime 60초, startup/handshake/drain 각 5초, 종료 grace 3초,
동시 group 프로세스 3개, input 64 MiB·output 2 MiB·runtime 128 MiB·전체 256 MiB다.
외부 wrapper는 각 프로세스에 상속되는 hard/soft 주소 공간 2 GiB를 설정했다.
고유 입력은 30,736,948 bytes였고 감독된 트리의 관측 byte 수는 114,972,764다.
프로세스/트리 관찰을 hard child/disk quota나 전체 peak RAM 계측으로 보고하지 않는다.
Fastchess의 per-move 500ms와 내부 read margin도 공통 whole-engine 시계 인수를 대신하지 않는다.

수 제한으로 끝난 원시 PGN의 draw는 A 기반 감사에서 **Incomplete 두 판**으로 보존했다.
`scored_games=0`, `strength_eligible=false`, `execution_ready=false`다. D의 총 1,032 완료는
수락한 평가 응답의 프로세스 집계이며 물리 추론 호출 수나 모든 go/root의 event journal이 아니다.
LC0 비교·탐색/runtime A/B·Elo 개선으로 해석하지 않는다. 목표 GPU·GPU 종단 계측·실제
학습·정식 paired 강도 실험은 각각 pending이다.

실패도 보존했다. `7e7067a` CI의 common MSRV 불일치, `95932e2` 로컬 clippy의
unused wrapper/큰 오류 값, `57bcc65` 로컬·Linux CI의 O_PATH fchmod 실패와 Windows
CI의 non-Linux 타입 추론 실패를 원래 로그에서 확인한 뒤 수정했다. 권한을 생략하거나
MSRV를 올려 통과시키지 않았다. 준비 실패는 원래 cause와 부분 복사·writer 종료·저장
오류를 구분한다. postspawn ownership loss에서는 원래 Child와 입력 owner를 함께 보존하고
입장을 닫으며, 보존을 PID signal/reap 권한 회복으로 표시하지 않는다. 문서 후속과 develop
인수에서는 `0f0d70d`의 실행 영향 입력과 실제 CI 대상 SHA를 다시 대조한다.

## 첫 목표 CUDA 수치·provider 인수

[PR #13](https://github.com/daejunnom/RoveZero/pull/13)의 실제 실행 소스는
`bde687c7f269af3c7cd501201edde14c5c8ef642`이며 공통 계약 revision 0.1을 유지한다.
총괄은 C의 runtime pin·물리 worker·오류와 B의 native attestation을 수동 정합했다.
새 `RuntimeLibraryLoad` cause를 B wire에 전달하고, 완료 미확정 CUDA 오류에서는
원래 cause·session·입력·job을 격리하며 Ready/ActualCompute를 만들지 않는다.
`rz-native-loader`는 검토한 ORT 3개·NVIDIA 16개의 exact name/bytes/SHA 실행 profile,
dependency-first load, process latch, retained FD/handle와 실제 inode mapping을 소유한다.
CPU pin·기존 CPU backend identity와 `rz-eval`의 `forbid(unsafe_code)`를 보존한다.

| 실제 검사 | 결과와 적용 범위 |
|---|---|
| Local workspace test | `e95ba37`: 626 passed, 0 failed, 16 ignored, 63 test binaries |
| Local fmt·workspace clippy | `e95ba37` 소스 통과; strict `-D warnings` |
| Rust 1.85 검사 | loader·eval `--features onnx --locked` 통과; contracts의 A 의존은 기존 Rust 1.90 요구이므로 all-features 1.85 지원으로 보고하지 않음 |
| 최종 소스 CPU CI | [run 37079578861](https://github.com/daejunnom/RoveZero/actions/runs/37079578861), `bde687c`, Ubuntu·Windows의 fmt/test/심화 규칙/Python fixture/clippy 모두 성공 |
| 실제 목표 장치 | RTX 4050 Laptop GPU, 6141 MiB, driver 610.62, Linux x86_64; device 0 |
| 실제 CUDA 실행 | FP32, TF32 off, CPU fallback 금지; warm profile의 CUDA kernel 이벤트 98개, CPU 0개 |
| 독립 참조·batch | 원본 LC0 Eigen protobuf 참조 12개, batch 1/2/4/8/16 및 역순 대조 통과 |
| 최대 절대 오차 | CUDA raw logit `1.5735626220703125e-5`, 합법 policy `8.940696716308594e-7`, WDL `4.172325134277344e-7`; batch logit `1.5139579772949219e-5` |
| 입력 경계·C worker | empty/과대 batch/NaN 거부 후 정상 실행; fixture Rules의 물리 출력 4개·취소 거부 1개 |
| 최종 CPU 회귀 | 같은 release binary·기존 CPU core·같은 참조에서 통과; GPU 파일을 CPU로 조용히 대체하지 않음 |
| 종료 | 두 gate exit 0, 남은 cgroup PID 없음, 전용 cgroup 제거 확인 |

기존 원본·ONNX·manifest·12개 참조를 hash로 잠가 재사용했다. 첫 bundle은 ORT GPU
1.22.0, CUDA 12.8 계열, cuDNN 9.8.0.87이며 native 19개 합계는 2,970,143,952 bytes다.
전체 canonical bundle SHA는 `12d8fe080ab22c29d9a63229f233066ef0bd1f6b4743a2d54d1be9efea20fcd3`,
GPU backend SHA는 `fa724b400eaa5e3c632de51be76763c4d29b363f4261c1c87b044daad2369b36`다.
CPU backend SHA는 `90724c106ce7545e1d90a8ff12c5f837f92d049ea6948dc7c61f766bc64ecb78`이며
이는 이 검사의 batch 16 설정이다. 기존 UCI batch 1 backend와 동일하다고 표시하지 않는다.
독립 검토자가 manifest와 profile codec으로 두 backend SHA를 재계산하고 spec/source/
README/준비 receipt의 19개 name/bytes/SHA 일치와 실제 profile 내용을 대조했다.

| 외부 산출물 | bytes | SHA256 |
|---|---:|---|
| 최종 `maia_check` binary | 1,811,968 | `cb803553f9b6c8667db452ea6c1b69e0cbe1d42c4476934eb0022de5f0f05af8` |
| release build receipt | 632 | `5c3042f72b4dda4341258964c409e65658f0a6ef12a01d9a8833c035ce6ae7f9` |
| CUDA numerical report | 8,089 | `1c1b0207c7dbd3b4b994ff20ea403dacee24c2aa6595725ee0b94ed8541b6172` |
| CUDA execution receipt | 3,846 | `9a52f79acac78ccdde9ff8063b3690a9aad643c913ce75a4252600eb477c3bec` |
| CUDA warm profile | 41,261 | `c3c4e0d16080a4f4142605c3366bb1a1141bf2609f9dd11478dfc14fe55e3671` |
| CPU numerical report | 4,127 | `67855aab943047e06f2f6479b2ffd48e22c80523a2b4c2a4159ae2917e43451d` |
| CPU execution receipt | 3,432 | `73fb8eedebba7dcfa1ea4a4e887e7daceea328d3b8507a94c82d9811c3a371fd` |

원시 자료의 논리 root는 저장소 밖 `reports/coordinator-integration/native-gpu/`다.
최종 `captured-bde687c/`에 41개·304,588 bytes를 회수해 길이·SHA를 각각 대조했다.
회수 receipt SHA는 `a58a1739559c80e689922823fe0fde6693ff5a39c5b4c496b99bf29ac54cad8a`다.
이 별도 회수에는 profile·stdout/stderr·adapter 표본·실패·검사·build 기록을 포함하며
큰 native library·binary·가중치는 넣지 않는다. 준비 자료 8개·103,079 bytes도
`captured-preparation-v2/`에 별도 보존한다. library/notice와 Linux의 private copy는 유지한다.

외부 gate의 hard 제한은 wall 120초·cgroup memory 4 GiB·swap 0·pids 128·CPU 2 core·
file 1 GiB다. non-native file 8 MiB와 전체 run 4 GiB는 감독 관측 제한이며 disk quota가 아니다.
CUDA cgroup peak는 file cache 등을 포함하여 **4 GiB 상한에 도달**했고 max event 31,113,
OOM/kill 0이었다. sampled RSS는 998,096,896 bytes로 cgroup peak와 다르다.
100ms 전체 adapter 표본은 123개·10~214 MiB였으며 다른 client를 포함한다. 이를 process
peak VRAM이나 관측 사이의 실제 peak로 보고하지 않는다. CUDA 13.1829초·CPU 0.4122초는
서로 다른 크기의 pin copy·hash·warmup·검사를 포함한 gate 시간이며 추론 속도 비교가 아니다.

첫 e95ba37 시도의 8 MiB hard file 제한이 library copy를 SIGXFSZ로 종료한 실패와,
최종 binary에 이전 SHA를 전달하여 native 실행 전에 거부된 실패를 보존했다. 감독 조건과
binary pin을 명시적으로 수정한 뒤 fresh namespace로 실행했고 오차·precision·batch를 바꾸지 않았다.
e95ba37 Windows CI는 redundant profile-builder binding의 clippy 실패였으며 해당
non-Unix 두 줄만 제거하고 최종 bde687c의 두 OS CI와 GPU/CPU 실행을 다시 관측했다.

인수 범위는 **선정 Maia의 수치·provider probe와 C worker**다. 실제 A Rules→D scheduler→
B 소비/backup의 GPU 연결, GPU 취소·deadline·root 교체의 완료 수명, D02의 종단
P50/P95/P99·완전 event journal, process peak VRAM, D03 효과, 실제 학습·정식 paired
강도 판정은 pending이다. 파일 origin 감사도 exact mapped path/device/inode 범위이며
개별 심볼 결합이나 모든 OS/driver ELF closure를 인증하지 않는다. 자세한 실행·로더 경계는
[GPU-RUNTIME-BOOTSTRAP](GPU-RUNTIME-BOOTSTRAP.md)을 따른다.

## 실제 Rules·CUDA·D 수치 연결

[PR #14](https://github.com/daejunnom/RoveZero/pull/14)는 실제 A Rules→C CUDA owner→
D scheduler→B UCI 소비와 E의 별도 GPU runner를 연결하는 후속 작업이다. 공통 계약은
0.1을 유지하며 CPU V1 wire를 CUDA까지 확장하지 않는다. C origin·D device 예약·
common delivery·B의 최종 guard 뒤 소비·E의 별도 DTO를 총괄이 수동으로 맞춘다.
이 절의 수치 gate와 아래 UCI/runner 인수는 서로 다른 실행 증거로 기록한다.

`f8b3ed5ea3f6fe7ed8f6fe68eb3b31c2daec6618`의 `rules_maia_check`로 실제 immutable
Rules 상태 12개를 기존 LC0 원본 참조와 대조했다. 원본 No 5개·RepeatOldest 7개의
입력과 출력 의미를 유지했다. 같은 binary의 명시 CPU 회귀도 별도 프로세스로 통과했다.
UCI 첫 제품 프로필은 No이며 이 예제의 지원 encoding 범위와 구분한다.

| 실제 로컬 검사 | 결과 / 범위 |
|---|---|
| C focused all-target/all-feature | 61 passed / 0 failed / 12 test binaries |
| D focused all-target/all-feature | 58 passed / 0 failed / 4 test binaries |
| B UCI/search focused all-target/all-feature | 245 passed / 0 failed / 13 test binaries; `16efe40` 관련 소스 |
| 실제 Rules→C CUDA→D | 12개 dense input exact, legal policy 최대 오차 `8.940696716308594e-7`, WDL `4.172325134277344e-7`; 각 허용치 `1e-4` |
| 같은 binary CPU 회귀 | 12개 dense input exact, policy `1.1920928955078125e-6`, WDL `4.76837158203125e-7`; 같은 허용치 |
| CUDA origin·placement | 각 history profile의 actual `cuda_onnx`, device 0·arena 1 GiB·B1; 새 profile 각각 CUDA kernel 98·CPU 0 |
| D 완료·종료 | 양 provider의 두 profile 모두 physical drained, execution/logical/host/device/pinned/request 예약 0, unexpected delivery·diagnostic 없음 |
| 감독 종료 | 양 gate exit 0, 남은 전용 cgroup PID 없음, cgroup 제거 |

예제 binary는 2,095,992 bytes,
SHA `1d5459bdb677f10d48f1ebc7885c45b43bc9834bf81c4b168f57b20bd6aa3f22`다.
CUDA report SHA는 `f9770dd69632793d591e8230f310699a3302d6867bc58349759c4d7b2950ded1`,
execution receipt는 `3713afc51b390a67870dac89d10a6ef5bcd3c20effc2c0ed5e83d3826df793e9`다.
CPU report SHA는 `4b18dcdecbeaf333079c80c985cdbd8fba869eb0e9bdbd3319e67cafa822e6c2`,
receipt는 `ed3cfd5945ddf121d99ac6759c5135c38b590e4d81042bba12ed4861c882a8f1`다.
새 CUDA placement의 실제 digest 두 개를 report와 독립 대조했으며, 이전 warm 파일을
재사용하지 않았다. 작은 자료 14개·125,991 bytes를 저장소 밖
`reports/coordinator-integration/native-gpu/captured-rules-f8b3ed5/`로 회수해 길이와
hash를 각각 확인했다. 회수 receipt SHA는
`bef0fd36cddd71cfdb2f4f257391cbfd0f1bbd39c2d4935ec7fdad62d0b31472`다.
큰 native copy·weights·binary는 이 회수에 넣지 않고 Linux의 원본과 실행 사본을 보존한다.

gate 상한은 wall 120초·cgroup RAM 4 GiB·swap 0·pids 128·CPU 2 core다.
CUDA cgroup peak는 **4 GiB 상한에 도달**, max event 30,711·OOM/kill 0이었고 sampled
RSS는 940,417,024 bytes였다. CPU cgroup peak는 96,051,200 bytes다. CUDA 16.134초와
CPU 0.387초는 pin copy·hash·bootstrap·수치 검사를 포함하며 추론 속도 비교가 아니다.
native 단일 파일 1 GiB·non-native 8 MiB·run 4 GiB의 관측 제한과 filesystem quota를
구분한다. 100ms 전체 adapter 표본과 admission 선언은 process peak VRAM 증거가 아니다.

독립 검토는 report·receipt·새 placement 7개, 111,379 bytes의 내용과 참조 digest를
대조했다. binary/native byte를 재hash한 검토는 아니며 TF32 off·thread 1·fresh full-step과
19개 세부 member는 같은 SHA의 source/closed profile 근거를 함께 사용한다. warm profile의
model_run 하나와 case별 D 완료 12개는 별도 사실이고 전체 physical journal로 합치지 않는다.

첫 WIP CI `37082982929`의 잠금 파일 누락은 `16bb00e`에서 수정했다. 후속
`37083533259`는 Ubuntu·Windows 모두 CPU V1의 CudaOnnx 미처리 match에서 실패했으며,
원래 job 로그를 보존했다. `16efe40`에서 typed CPU projection 거부와 CUDA 전용 DTO로
정정하고 focused B 검사를 통과했다. 이 수치 gate만으로 B 소비/backup·GPU stop/
deadline/root 교체·E runner의 인수를 주장하지 않는다. D02 종단 journal·process peak
VRAM·D03 효과·실제 학습·정식 강도 비교도 별도다.

## 실제 CUDA UCI 소비·합법 fallback·유한 회복

`bd3e4e17b163b4b7b0917b79b07f0e73e2974eab`에서 Rust 1.96.0의 전체 workspace
all-target/all-feature 검사는 **656 passed / 0 failed / 16 ignored**, 기본 native CLI는
4 passed였으며 fmt와 strict clippy도 통과했다. CUDA UCI와 E arena의 release feature를
명시해 빌드했다. CUDA는 별도 startup/final V1 DTO·파일명을 사용하고, CPU V1의 닫힌
필드 집합에 CUDA 증거를 추가하지 않는다. 실제 CPU 회귀 receipt도 이를 대조했다.

같은 source의 `rz-uci`는 2,949,408 bytes,
SHA `eb9849946a8d9c8ae992eadd81fdf9a22a8bcb7f1d9265a1e030bf820b75718f`다.
실제 CUDA gate receipt SHA는
`3ee013e3e8c705620c12cdf5e5c78e9a0f98eb491aa1b89ec075703f476f815b`, CPU 회귀는
`2acd02cc5ec1af24932b1be5b509b5ecaee997f2647fdd297ee8bbb3cfc136b0`다.
새 CUDA placement trace의 실제 SHA는
`d93d6e3ebcb4f98bcf225d6acc66bf5e36d7422c072d06acb1ecaa2a75c4b134`이며,
warm probe의 Node 98개는 모두 CUDA, CPU Node는 0이었다. 이 warm trace를 D 완료
횟수나 모든 physical inference 호출의 journal로 해석하지 않는다.

| 실제 프로세스 검사 | CUDA | 명시 CPU 회귀 |
|---|---|---|
| 독립 opening fixture의 합법 착수 | 5개, duplicate bestmove 없음 | 같은 5개, duplicate 없음 |
| D normal-poll computed 완료 | 45 | 51 |
| B 최종 tree guard 뒤 소비 | root 초기화 4, non-root backup 41 | CPU V1에는 이 CUDA 전용 필드를 직렬화하지 않음 |
| stop 직후 새 게임의 네 번째 착수 | `WorkerLimit`의 합법 fallback `a7a5` | 같은 busy fallback |
| 별도 회복 단계 | 고정 250ms 뒤 단 한 번의 go, game 3/root 12의 computed 완료와 guarded non-root backup | 같은 game 3/root 12의 computed 완료 |
| 종료 | exit 0, confirmed physical drain, original/collection/mapping failure 없음, 관측 손실·overflow 0 | exit 0, confirmed drain, original/collection failure 없음 |

실제 transcript는 `uci/isready`, 흑백 opening 상태, `go infinite`, 연속 stop 두 번,
`ucinewgame`, quit를 포함한다. 네 번째 수의 busy fallback과 Admission/Stale 진단을
stderr에 보존한다. 다섯 번째 수의 실제 새 game/root 문맥으로 회복을 확인했으며,
250ms 대기나 `isready` 응답 자체를 physical drain 증거로 쓰지 않았다. 모든 root의
일대일 journal, 강제로 배치한 physical race, 모든 취소/deadline 조합의 실제 GPU
검증을 완료했다고 보고하지 않는다. 기존 `6d340ea` v1의 네 번째 CUDA fallback과
회복 미입증 결과도 원본 그대로 보존한다.

외부 gate는 wall 120초·cgroup RAM 4 GiB·swap 0·pids 128·CPU 2 core·단일 파일
1 GiB로 제한했다. run 4 GiB·non-native 8 MiB·stdout/stderr 합계 2 MiB의 감독 관측을
구분한다. CUDA cgroup peak는 **4 GiB 상한**, max event 30,663·OOM/kill 0이었고
sampled RSS는 940,212,224 bytes였다. observed kernel VmPeak는 38,827,732,992 bytes다.
이 UCI gate에는 RLIMIT_AS를 설치하지 않았으므로 후속 128 GiB 제한의 실제 성공으로
승격하지 않는다. CPU cgroup peak는 94,617,600 bytes다. 두 gate는 남은 cgroup PID 없이
종료하고 전용 cgroup을 제거했다. 전체 adapter의 100ms 표본은 process peak VRAM이 아니다.

`37085303468`의 양 OS CI는 큰 `ContractPumpEvent` payload에 대한 strict clippy로
실패했다. `Option<Box<AcceptedEvaluation>>`와 명시 consumer borrow로 수정하고 tree
guard·wire 의미를 유지했다. 이어 E의 불필요한 borrow와 C 예제의 cohesive profile
인자 묶음을 수정했다. `37086216891`은 Ubuntu 성공·Windows의 Linux 전용 import
경고 실패였으며 해당 실제 job 로그를 보존하고 cfg 범위를 맞췄다.

최종 product source `e0d7e131548fe0d84bda9b38ef261adcc3390e79`의
[CI 37086498966](https://github.com/daejunnom/RoveZero/actions/runs/37086498966)은 Ubuntu·
Windows 모두 fmt·workspace·기본 CLI·독립 Rules·F fixture·strict clippy에 성공했다.
`bd3e4e1` 이후 source 차이는 E의 Linux 전용 import뿐이다. 같은 설정으로 다시 빌드한
UCI의 실제 bytes·SHA가 위 gate binary와 같음을 확인했으며 B/C/D/Rules·workspace·
가중치·backend 입력이 바뀌지 않았다. 재사용 확인을 새 GPU 실행으로 표시하지 않는다.
E의 현재 release binary는 별도로 빌드했으며 실제 GPU pair 인수는 후속 절에 기록한다.

독립 검토는 v2 helper·build·양 gate·startup/final·stdout/stderr·새 placement 13개,
95,488 bytes의 실제 내용을 대조했다. 별도 Windows 회수에는 adapter 표본과 최종
재빌드 기록도 포함하여 **16개·101,340 bytes**를 저장소 밖
`reports/coordinator-integration/native-gpu/captured-uci-bd3e4e1-v2/`에 보존하고 각 길이와
SHA를 확인했다. 회수 receipt SHA는
`4c1a4f12e680eb2ef919c62a0cc9199c5fd03db2ed36c762c0268e4270ec39e4`다.
큰 native library·모델·binary를 이 metadata 회수에 넣지 않았다. 원본 Linux gate와
private native copy는 보존하며, 이 회수는 새로운 추론 실행이 아니다.

## 첫 실제 CUDA pair 실패와 종료 경계

`e0d7e131548fe0d84bda9b38ef261adcc3390e79`에서 E의 CUDA 전용 lock과 실행 경로를
사용한 첫 실제 pair는 **실패**했다. 시작 상태·원본/ONNX·19개 bundle·FP32·device 0·
B1·thread/worker 1·HistoryFill No를 고정하고, 두 판 각각 최대 6 ply의 integration
cutoff를 사용했다. 첫 판의 실제 PGN은 6 ply를 담지만, 둘째 판은 시작 검사를 끝내지
못했다. 정식 완료 pair·득점·강도 자료로 사용하지 않는다.

첫 판의 baseline PID 402는 D computed 완료 258개와 B root 초기화 2개·non-root
backup 256개를 기록하고 exit 0으로 끝났다. Candidate PID 413은 각각 252·2·250개,
confirmed physical drain·mapping failure 없음의 최종 기록을 쓴 뒤, quit 약 36.75초
후 **SIGABRT 6 / status 134**로 종료했다. 최종 native 기록과 실제 process exit는
서로 다른 증거다. 이 실패 뒤 새 candidate PID 447의 uciok는 약 46.63초 걸렸으며,
baseline PID 455는 60초 startup deadline을 넘었다. 두 새 프로세스의 실제 검색
완료 수는 0이다. native abort의 직접 원인은 아직 확인하지 않았다.

E receipt는 runner exit 1과 검증한 owned-group cleanup을 보존하고 integration을
false로 판정했다. provider와 PGN 인수는 process gate 실패 때문에 명시적으로
미실행이며, provider sessions는 비어 있고 scored games는 0이다. 상위 CLI는 exit
2로 끝났다. 성공으로 남은 개별 native receipt를 부분 pair 성공으로 승격하지 않았다.
pair receipt SHA는 `6270b2c303056abd5357c68b7968163ede40ea072fcaab687df34190f9ea937a`,
PGN SHA는 `6dfe1ab14de116cbb21062da1e0e8682509f643f588f700e183d12478e714fb4`다.

감독 실행은 wall 300초·cleanup 10초·전용 cgroup RAM 8 GiB·swap 0·CPU 2 core·
pids 128·per-process address space 128 GiB·단일 파일 1 GiB로 제한했다.
실제 전체 시간은 228.208초, cgroup peak는 **8 GiB 상한**, max event 196,709·
OOM/kill 0이었다. sampled aggregate RSS는 1,801,588,736 bytes이며 kernel VmPeak
관측은 38,827,728,896 bytes다. anon/file-cache 구성과 PSI를 수집하지 않았으므로
이 수치만으로 abort 또는 startup 지연의 원인을 판정하지 않는다. run 16 GiB·
entry 256·non-native stream 8 MiB는 감독 관측 한도이며 filesystem quota가 아니다.
남은 cgroup PID·cleanup/preservation 오류 없이 종료하고 전용 cgroup을 제거했다.

실패 metadata 25개·426,423 bytes를 저장소 밖
`reports/coordinator-integration/native-gpu/captured-pair-e0d7e13-v2-failed/`로 회수하여
각 길이와 SHA를 대조했다. 회수 receipt SHA는
`a47f1fc347438f8cb73688d64d7d4782d62716bbb610f21922619cdc83726686`이다.
원래 로그·private copies·첫 실패 자료를 보존하고 자동 재시도하지 않았다.

소스 조사에서 확정한 결손은 `SingleWorker`의 JoinHandle을 보존하지 않아 요청의
physical drain 뒤 worker closure와 native session destructor의 종료가 최종 기록
밖에서 진행된다는 점이다. process-owned worker의 admission 종료·native teardown·
thread join을 유한하게 확인하는 경계를 추가하고, per-root drain 및 quarantine의
원래 pin 보존과 구분하여 재검증한다. join 추가만으로 SIGABRT가 해결됐다고
주장하지 않는다. 이 접점과 실제 pair process gate는 PR 인수 전 남은 검사다.

후속 source는 C의 단일 reaper와 B final join 조건, E의 각 native PID 종료 TRACE
검사를 연결했다. 공통 revision 0.1·CPU/CUDA V1 키 집합·per-root session 재사용·
기존 final 2초와 pair 자원 한도는 유지한다. 같은 수정 tree의 실제 로컬 workspace
all-target/all-feature 검사는 **673 passed / 0 failed / 16 ignored**, 기본 native CLI는
4 passed이며 fmt와 strict clippy도 통과했다. 새 의미 검사 17개는 destructor/TLS
지연·생성 실패·panic·Q·admission·root 재사용과 네 PID의 종료 기록을 검사한다.
이는 CPU/mock source 인수이며 수정 후 실제 CUDA UCI/pair는 별도 실행으로 기록한다.

## worker 종료 보완 뒤 실제 CUDA pair 인수

실행 source는 `e45dd1050d8782ff4ae3e524341cdbac52209cf5`이며,
[CI 37089717057](https://github.com/daejunnom/RoveZero/actions/runs/37089717057)은
Ubuntu·Windows 모두 성공했다. 같은 깨끗한 tree에서 CUDA feature를 명시해 release
빌드하고 source·root lock·Rust 1.96.0·명령·binary를 외부 build receipt로 고정했다.
WSL의 Git은 Windows worktree gitfile을 해석하지 못하므로 source HEAD와 clean 상태는
Windows Git에서 snapshot 전후 직접 확인했다. 이는 source/binary 출처 확인이며
기존 binary를 새로운 source의 산출물로 재명명한 결과가 아니다.

| source e45dd10 산출물 | bytes | SHA-256 |
|---|---:|---|
| CUDA/CPU UCI binary | 3,001,096 | `d94ca3ab481335dd1414d3cb8465d97b67a99126fa96fb5f950818d1f6d05c3f` |
| CUDA arena binary | 4,561,000 | `64f23a415a3cd87f2c74a03264b535cdc8cd93bd7ae92c32d881e890b127c9dd` |
| 실제 build receipt | 1,244 | `1e7ce6dbea9733a12d8f1beece250a6b81b948a0a28363783cefa3a329716a9b` |

새 실제 UCI v2 gate는 CUDA와 명시 CPU 모두 exit 0·confirmed drain·원래 오류 없음으로
끝났다. 각각 다섯 opening fixture 착수는 `e2e4, c7c5, e2e4, g8f6, g8f6`이다. 이번
stderr에는 Stale 진단이 있지만 WorkerLimit/fallback 진입 근거가 없으므로 과거 v2의
네 번째 busy fallback 결과를 이번 실행으로 옮기지 않는다. CUDA D computed 완료는
54개, B guarded root 초기화 5개·non-root backup 49개다. CPU D 완료는 61개다.
마지막 문맥은 game 3/root 12이며, CUDA drain-discarded result 1개와 observation
손실 0·overflow false를 구분한다. 모든 결과가 탐색에 소비됐다고 주장하지 않는다.
새 CUDA warm trace는 CUDA kernel 98·CPU 0이며 실제 SHA는
`2cd6c956ab8a2cbf9d8f9cd8eb03d76df45fb48f7dcf71d1331bf33d8409e01c`다.
CUDA gate SHA는 `c733eec25df2467e2d917d036da68f1fd0fa96f758eefc4a2f8fe38a4473d4bd`,
CPU gate SHA는 `164148d94a85c4e8cae83005753b8576bbc46635692f07488ec642fa2b04c2de`다.
UCI metadata 15개·100,240 bytes를 Windows 외부 root로 회수하고 각각 hash를 확인했다.
회수 receipt SHA는 `9fea0af9e326937f7731a6c6b32180164a47cbb12b819efa0a5d9dbee932a388`다.

같은 source의 새 E attempt `pair-e45dd10-v1`은 첫 실패와 다른 namespace에서 **단 한
번** 실행했고 실제 인수에 성공했다. 가중치·bundle·FP32·TF32 off·device 0·B1·thread/
worker 1·HistoryFill No·같은 시작 상태를 유지했다. 준비·hash·copy·bootstrap·runner·
최종 검증·cleanup을 포함한 전체 시간은 **199.576초**, Fastchess 자체 실행은
174.767초다. 두 숫자를 추론 지연이나 D03 속도 효과로 사용하지 않는다.

| 실제 native PID | 역할 | D computed | B guarded root / non-root | 종료 |
|---:|---|---:|---:|---|
| 369 | baseline, 첫 판 | 258 | 2 / 256 | final 정상, 실제 exit 0 |
| 392 | candidate, 첫 판 | 258 | 2 / 256 | final 정상, 실제 exit 0 |
| 414 | candidate, 둘째 판 | 258 | 2 / 256 | final 정상, 실제 exit 0 |
| 442 | baseline, 둘째 판 | 258 | 2 / 256 | final 정상, 실제 exit 0 |

네 startup/final의 loaded origin·binary·모델·encoding·backend·19개 bundle·새 placement
digest를 맞췄다. 네 프로세스 모두 실제 CUDA 추론과 Rules search 소비를 기록하고,
original/collection/mapping failure·report failure·overflow 없이 physical drain과
session/worker join을 확인했다. 각 새 trace의 CUDA Node는 98개이며 CPU Node는 0이다.
이 warm trace와 D 총 1,032개·root 8개·non-root 1,024개의 process aggregate를
전체 physical inference journal이나 모든 root의 일대일 성능 기록으로 합치지 않는다.
네 PID 각각의 고정 Fastchess 종료 TRACE status 0을 source gate와 별도로 확인했다.
runner exit 0·CLI exit 0·Gone·pending child 없음·unresolved owner 없음도 확인했다.

PGN은 동일 startpos의 `e2e4 e7e5`를 포함해 두 판 각각 6 ply이며 baseline/candidate의
흑백 배정이 바뀐다. A 감사에서 두 판 모두 **Incomplete**로 분류하고 scored games는
0이다. `integration_checks_passed=true`와 `execution_ready=false`,
`strength_eligible=false`를 함께 유지한다. 정식 LC0 비교나 승률 개선의 증거가 아니다.
E receipt SHA는 `c53c0c81039ca8cad21d71abbaad90c666e968d9edaea22fbfdfe5b756cc5095`,
PGN SHA는 `5a913de0c1826065d0d7a715d8b9e42a41f1687d667264bccad4ceb3f4cd2726`다.

외부 helper v2는 기존 v1의 자원·실행 통제를 유지하고 종료 시점의 owned cgroup
`memory.current/stat/pressure` 조회만 추가했다. helper SHA는
`a5fc34ef0b12d218f0270fdabf663a5c379586478e3bbddf8dcd078725e62007`이다.
동일한 wall 300초·cleanup 10초·RAM 8 GiB·swap 0·CPU 2 core·pids 128·per-process
AS 128 GiB·단일 파일 1 GiB 상한을 실제 적용했다. cgroup peak는 8 GiB 상한,
max event 197,004·OOM/kill 0이고 sampled aggregate RSS는 1,800,077,312 bytes다.
종료 시점 file charge는 8,512,729,088 bytes·anon 0, memory pressure full 누적은
4,458,942 microseconds다. 이는 종료 시점/누적 관측이며 최초 SIGABRT의 원인이나
전체 시간대별 메모리 구성의 증거가 아니다. native destructor 보완 뒤 이 한 번의
실행에서 abort가 없었다는 사실과 근본 원인 확정·모든 종료 조합 검증을 구분한다.

전용 cgroup의 잔여 PID와 original/cleanup/preservation 오류 없이 종료하고 cgroup을
제거했다. 성공 metadata·recipe·helper·build 자료 **31개·511,927 bytes**를 저장소 밖
`reports/coordinator-integration/native-gpu/captured-pair-e45dd10-v1/`에 회수하여 각
길이·SHA를 확인했다. 회수 receipt SHA는
`af7d694e1fa63f29f57f7870b41723117e261bc3b024b8bcc89729c223428c21`이다.
첫 실패와 성공의 raw PGN·receipt·로그·private copies는 별도로 보존한다.

별도 담당의 읽기 전용 감사에서도 UCI 두 gate·E의 작은 receipt/log/placement 자료,
16개 retained artifact의 실제 bytes/hash, locked spec→26개 snapshot의 ArtifactRef→네
startup/final·PID·TRACE를 대조해 제한된 연결 인수의 차단 사항을 찾지 못했다.
이 감사는 binary·모델·19개 library 원시 bytes의 독립 재해시나 새 GPU 실행이 아니다.
가장 느린 `uci`→`uciok`는 53.899초로 60초 한도 안에 들었고, `quit`→실제 종료 TRACE는
0.701~1.302초다. TRACE 간격을 worker join 함수만의 측정값으로 사용하지 않는다.

D02 GPU 종단 source clock/journal·P50/P95/P99·process peak VRAM, D03 단일 runtime
개선, 실제 Maia F02 학습, 정식 LC0 paired 강도·통계는 여전히 별도 인수다.

## 최적화 소스 b5ba853의 로컬 CPU·CUDA 연결 인수

2026-10-03 총괄 I02는 `develop`의 깨끗한 product source
`b5ba853585cb5a78f81159f733a86cdfe085936b`에서 CPU 수치 회귀, 실제 A Rules→C CUDA→D,
CPU/CUDA UCI와 E의 제한된 CUDA pair를 인수했다. 원격 open PR이 없음을 먼저 확인했다.
[PR #16](https://github.com/daejunnom/RoveZero/pull/16)의 의미 보존 최적화를 포함한 소스이며,
이 절 이후의 문서 head와 실행한 product SHA를 구별한다. 공통 계약은 `0.1`이다.
실행 환경은 Ubuntu/WSL2·Rust 1.96.0·RTX 4050 Laptop 6GB·driver 610.62다.

root Cargo.lock SHA는
`e86e6deca01f9f5d74a11014cd5473dd98f6759a64540ca26377a6cb44c7b57d`다.
Maia-1900 원본·독립 LC0 v0.32.1 Eigen reference·변환 ONNX·ORT 1.22.0을 유지했다.
CUDA는 FP32·TF32 off·device 0이며 Rules/UCI/pair는 B1·thread/worker 1을 사용했다.
canonical native bundle은 19개 파일,
`12d8fe080ab22c29d9a63229f233066ef0bd1f6b4743a2d54d1be9efea20fcd3`다.

| b5ba853 release 산출물 | bytes | SHA-256 |
|---|---:|---|
| C 수치 검사 | 1,827,936 | `ffdb36684749118578ced2f05380ef0886ed2f1ef3d55c15823d00bde6247220` |
| 실제 Rules/C/D 수치 검사 | 2,104,584 | `36158caa6c5d14a79e48cfca27bb382411c17f5c47d3231832593ee10c3cb410` |
| CPU/CUDA UCI | 3,002,248 | `991f6746c9f9c54c1a97bad37dfe6df59a0652ea4e9981ba94d789578fc5840d` |
| CUDA arena | 4,562,000 | `0037315239200a9b2fb923576287a233d71a729662a20765668fc79f765cbcd5` |

| 검사 | 이번 인수 근거 | 결과 |
|---|---|---|
| C 실제 CPU 수치 회귀 | 새 실행, 12 case·B1/2/4/8/16·LC0 reference 대조 | passed, 전체 gate 0.373초 |
| A Rules→C CUDA→D | 새 실행, 12 case·No/RepeatOldest 두 profile·고정 projection과 D finalization | passed, 전체 gate 15.166초 |
| CUDA UCI | 새 실행, opening fixture 착수 5개·중복 stop·새 game·후속 실제 NN 계산·quit | passed, 전체 gate 14.282초 |
| CPU UCI | 같은 binary의 명시 CPU 경로를 새 실행, 같은 프로토콜 검사 | passed, 전체 gate 0.670초 |
| E CUDA pair | 새 실행 `pair-b5ba853-v2`, 두 판·각 6 ply·흑백 교환·네 새 프로세스 | passed, 전체 gate 150.642초 |
| C CUDA raw logits·batch | 같은 b5ba853의 09:01:32 UTC 실행을 입력/binary/report/profile hash로 재사용 확인 | 기존 passed, 새 GPU 실행 아님 |

CPU legal-policy 최대 절대 오차는 `1.1920929e-6`이다. 실제 Rules/CUDA의 dense 입력 오차는
0, legal-policy/WDL 최대 절대 오차는 각각 `8.9406967e-7`/`4.1723251e-7`로 `1e-4` 이내다.
두 Rules profile은 각각 CUDA Node 98개·mapping 감사 성공·D 잔여 예약 0·physical drained를
기록했다. 실제 Rules projection 검사와 fixture handle의 소유 범위는 production registry의
모든 발급 경로 검증과 구분한다. 재사용한 C CUDA 검사는 12 case와 B1/2/4/8/16을 포함하며,
raw-logit 최대 절대 오차 `1.5735626e-5`, single/batch 차이 `1.5139580e-5`다. 보고서와
placement의 실제 bytes/hash를 다시 대조했고 CUDA Node 98개·CPU Node 0개를 확인했다.

두 UCI 실행의 합법 착수는 `e2e4, c7c5, e2e4, g8f6, g8f6`이며 중복 bestmove는 없다.
CUDA D normal-poll computed 완료 53개와 B 최종 tree guard 뒤 root 초기화 5개·non-root
backup 48개를 확인했다. drain 과정에서 별도로 버린 결과 1개, observation 손실 0·overflow
false를 보존한다. CPU D 완료는 61개이며 CUDA 전용 report 키를 CPU V1에 추가하지 않았다.
마지막 실제 문맥은 game 3/root 12다. 고정 250ms 대기와 `isready` 자체를 drain 증거로
사용하지 않는다. 두 실행은 exit 0·confirmed physical drain·원래/collection/mapping 오류
없음으로 끝났다. 모든 root의 journal이나 강제로 만든 모든 취소 race를 검증한 결과는 아니다.

E 첫 `b5ba853-v1` 감독 시도는 **프로세스 시작 전에 실패**했다. Cargo release 파일의
하드 링크 수가 2라 단일 링크를 요구하는 기존 pin 검사에 걸렸으며 cgroup이나 GPU 대국을
시작하지 않았다. 원래 기록을 보존하고 안전 조건을 완화하지 않았다. 같은 bytes/hash의
단일 링크·readonly 실행 파일을 별도 소유 경로에 복사한 뒤 새 input/lock/attempt v2를
준비했다. native 가중치·library pin과 복사 검증 조건은 유지했다.

| 새 native PID | 역할 | D computed | B guarded root / non-root | 실제 종료 |
|---:|---|---:|---:|---|
| 380 | baseline, 첫 판 | 227 | 2 / 225 | TRACE status 0, confirmed drain |
| 391 | candidate, 첫 판 | 243 | 2 / 241 | TRACE status 0, confirmed drain |
| 417 | candidate, 둘째 판 | 258 | 2 / 256 | TRACE status 0, confirmed drain |
| 431 | baseline, 둘째 판 | 251 | 2 / 249 | TRACE status 0, confirmed drain |

네 startup/final·binary·모델·bundle·placement SHA와 실제 Fastchess 종료 TRACE를 직접
대조했다. 각각 warm trace의 CUDA Node는 98개·CPU Node는 0개다. process aggregate
D 완료 979개·root 8개·non-root 971개와 손실 없는 관측을 전체 physical inference journal로
표현하지 않는다. retained artifact 16개·276,771 bytes의 실제 길이와 hash도 대조했다.
runner는 exit 0/Gone, CLI도 exit 0이며 pending child·unresolved owner·남은 cgroup PID가 없다.

PGN 두 판은 같은 시작 상태와 `e2e4 e7e5`를 포함해 각 6 ply이며 두 엔진의 흑백 배정이
바뀐다. A 감사에서 모두 **Incomplete**, scored games는 **0**이다.
`integration_checks_passed=true`, `execution_ready=false`, `strength_eligible=false`를
함께 유지한다. 동일 모델·탐색의 연결 검사이며 LC0 강도 비교나 최적화 GPU A/B가 아니다.
E receipt SHA는 `dba8472f11638bb0dfd68a20dae0388b87ed911243d07426767aee828b35a93b`,
PGN SHA는 `8e29d6fe78f0a19862cfc229dc28773553b9c87fac53bf883bbc5e62b2884449`다.

E 감독은 wall 300초·cleanup 10초·aggregate RAM 8 GiB·swap 0·CPU 2 core·pids 128·
per-process AS 128 GiB·단일 파일 1 GiB를 실제 적용했다. run 16 GiB·entry 256·depth 6·
로그 8 MiB는 감독 관측 한도다. cgroup peak는 8 GiB 상한, max event 196,989·OOM/kill 0이다.
sampled aggregate RSS는 1,800,036,352 bytes, kernel VmPeak 관측은 38,827,769,856 bytes다.
종료 시 file charge 8,512,806,912 bytes·anon 0을 보존했으며 cgroup peak를 live RAM이나
VRAM peak로 해석하지 않는다. 실제 owned output은 14,861,208,917 bytes·135 entries였다.
Fastchess 자체 시간은 124.739초다. 전체 gate 시간과 이 숫자 모두 추론 지연·D03 효과가 아니다.
각 단일 수치/UCI gate는 별도 wall 120초·RAM 4 GiB 한도로 끝났고 OOM/kill·잔여 PID가 없다.
모든 전용 cgroup을 제거했으며 최종 확인 때 `rz-uci/rz-arena/fastchess` 프로세스가 남지 않았다.

metadata 74개·850,554 bytes와 원본 PGN 1개·1,656 bytes를 Windows 저장소 밖
`reports/coordinator-integration/local-gpu-tests-20261003/`로 회수하여 각각 bytes/hash를
대조했다. inventory와 두 export receipt를 별도로 보존한다. metadata 회수 receipt SHA는
`ae8e0111a6f56912af10442168f40f18c3b1ec82699443f9cc9bbb5ec638c4ac`,
추가 PGN 회수 receipt SHA는 `c2ee5f8e08cdac83b7658551146956ed79d93f22089f7aae3bbb38e28d9f2e5e`다.
Linux 원본·첫 pin 실패·native private copies도 보존한다. 회수는 새 추론 실행이 아니다.
원시 로그·PGN·모델·native library·개인 경로는 Git에 넣지 않았다.

D02 source-clock/physical journal·요청별 P50/P95/P99·process peak VRAM, D03의 통제된
GPU A/B, 실제 Maia F02 학습과 정식 LC0 paired 강도·통계는 미실행으로 유지한다.
이번 로컬 연결 인수를 RunPod 환경 인수나 기존 ignored GPU 검사 전체의 성공으로 옮기지 않는다.

## 로컬 D02 host-source journal 인수

2026-10-03, 총괄 TASK-I02가 `88ea26e714a56af81f43cd1180c31df08fe813e9`의 native
CPU/CUDA profile off/on 네 실행을 인수했다. 공통 계약 0.1·W0/S0·FP32·B1·worker/thread 1과
CPU/CUDA V1 receipt key/type topology를 유지했다. B/C hook, D의 실제 원래 clock origin,
execution ID·backup guard·UCI 출력·drain 접점을 수동 대조했다.

각 실행은 같은 두 opening root의 128회 non-root backup, D computed 258개, 합법 착수
`e2e4, c7c5`, exit 0·confirmed drain을 기록했다. CUDA B의 guarded root/non-root는 2/256이다.
profile on은 각각 원래 timestamp의 journal 3896개·실제 물리 시도/완료/전달/소비 258개이며
누락·중복·identity/시각 불일치·unconfirmed·미소비는 0이다. 같은 소스의 workspace 검사는
682 passed·0 failed·16 ignored, fmt/strict clippy와 Ubuntu/Windows CI가 통과했다.

`--profile`은 8192 metadata record·JSON 8MiB로 제한되며 passive observation 실패를 engine
권한으로 삼지 않는다. startup placement warm-up 이후의 탐색 요청에 대한 journal이다.
CUDA 준비→backup P50/P95/P99는 3.371108/4.621393/5.615506ms, synchronous ORT Run host
interval은 2.349358/3.751160/5.041684ms다. 서로 중첩되는 단계의 분위수를 합산하지 않는다.
실제 PID의 VRAM 질의는 `[N/A]`이고 device Kernel/transfer trace도 없어서 미측정으로 남긴다.

73개 원본 metadata/journal/log·11,917,030 bytes를 저장소 밖으로 회수하고 bytes/hash를 확인했다.
[D02 상세 기록](research/LOCAL-D02-PROFILE.md)에 인수·phase 분포·통제 조건·남은 항목을 묶었다.
같은 game의 두 root 사이에서만 input key 반복 72개를 관측했으며 raw cache의 효과나 새 방문으로
해석하지 않는다. 다음 D03 단일 실험은 provenance·receipt·cache-hit 소비 계약부터 맞춰야 한다.
이번 결과는 host-source D02 통과다. device timing·process peak VRAM·D03 통제된 GPU A/B,
실제 학습·정식 paired 강도·통계와 RunPod 환경 인수는 남아 있다.

## PR #17·#18 병합 준비 인수

2026-10-04 총괄 TASK-I02는 #17 D02 계측과 #18 OPT-01~12를 제품 소스
`a93569bedb802a4eb07f19b12241595715d21df1`에서 함께 검사했다. 각 독립 PR의 CI만으로
호환을 판단하지 않고 B/C/D/UCI 여섯 충돌 파일의 source hook·buffer 반환·batch owner·
signal·guard·오류·수명 접점을 수동으로 맞췄다. 공통 계약은 **0.1**을 유지한다.
실제 조정과 병합 방식은 [계약 인수 10장](CONTRACT-ADOPTION.md), 상세 source/영수증은
[최적화 기록 12장](research/PERFORMANCE-OPTIMIZATION-PLAN.md)에 있다.

| 범위 | 현재 확인한 근거 | 상태 |
|---|---|---|
| #17 수정 소스 `b830121` | CI 37158044723, Ubuntu 683 passed·16 ignored / Windows 629 passed·2 ignored | 두 OS 필수 step 성공; ignored 미실행 |
| #18 통합 소스 `a93569b` | CI 37158070152, Ubuntu 705 passed·16 ignored / Windows 651 passed·2 ignored | 두 OS 필수 step 성공; 실패 0 |
| C 실제 CPU 신경망 | `673f3e4`의 default/buffers/binding/combined, 각 12 독립 참조·B1/2/4/8/16 | passed; 통합 뒤 영향 source·asset·binary를 대조한 재사용 |
| A/C/D/B 실제 CPU UCI | 통합 소스의 기본 profile, raw cache 새 게임 reset, binding, 폭 4, 폭 4+cache | 다섯 모드 합법 착수·정상 종료·오류 aggregate 0 |
| D02 저장·종료 | 기본 profile의 physical/complete/delivered/consumed 각 66, complete timeline, confirmed drain | passed; 첫 10초 종료 실패·부분 JSON 보존 |
| OPT-00 실제 native runner | 같은 소스·모델·backend의 새 프로세스, startup/termination·17 Computed·합법 e2e4 | diagnostic; formal_acceptance=false |
| full serial witness | 두 history fill의 16MB 전체 출력 hash가 원본과 동일 | 의미 보존 대조; NN/GPU 성능 검사가 아님 |

기본 native CLI 4개, release Rules/perft/독립 oracle 50개, Python model 도구 105개와
runtime runner 12개, fmt·strict Clippy도 통과했다. 수치 대조의 C worker와 실제 UCI의
A/C/D/B 경로를 구분한다. 실제 UCI에서 요청한 nodes를 stdout에 없는 방문 수로
바꾸지 않는다. 실제 cache-hit 전체 계수와 S1 물리 batch 분포는 기존 V1 receipt 범위
밖이며 합성 계약 검사와 실험용 후속 계측을 구분한다.

#17의 파일 저장은 producer 종료 후 64KiB로 모아 쓰도록 보완했다. #18의 runner는
manifest pair를 native CLI의 `--flag=value`로 전달하도록 수정했다. D02 v1에서
raw/batch/비기본 I/O의 실행을 asset 로딩 전에 거부해 불완전 trace를 complete로
증명하지 않는다. native 실험 옵션은 명시적으로 선택하며 S1 기본 폭은 1이다.

코드는 **#17 → #18 순서의 develop 병합 준비** 범위다. 이 기록으로 실제 병합을
실행하거나 main을 갱신하지 않는다. 새 CUDA binding/Graph/batch/cache의 수치·수명·
VRAM, 통제된 RunPod A/B, S1 대국 품질, 실제 학습·정식 LC0 paired 강도는 미실행이다.
원시 근거와 실패·복구 명세는 저장소 밖의
`reports/coordinator-integration/pr17-pr18-merge-20261004/`에 보존한다.

## PR #17·#18 develop 통합과 로컬 내부 A/B

2026-10-04 사용자 통합·로컬 강도 평가 지시에 따라 #17, #18을 순서대로 병합했다.
#17 merge는 `0318e6fffd1840fd2f00ffcde0dac6a5a2372334`, #18 merge와 실제 엔진 소스는
`bdd8523a915404f333497440bc6e5cd8996d7df3`이다. 최종 develop tree와 검토한 #18 head
`d412f15`의 tree가 동일하며 main은 이번 작업에서 변경하지 않았다. develop CI
[37198014988](https://github.com/daejunnom/RoveZero/actions/runs/37198014988)의 Windows·Ubuntu
workspace·native CLI·release Rules 독립 대조·Python 도구·fmt/strict Clippy가 성공했다.

새 엔진 binary의 CUDA 독립 수치 대조 12개·B1/2/4/8/16와 실제 CUDA node 98개,
폭 1·4의 제한된 UCI·착수·종료를 로컬 RTX 4050에서 확인했다. 이는 수치·연결 인수이며
S1 대국 우위나 RunPod 인수가 아니다. 공통 계약 0.1과 Computed-only native V1 B1
경계를 유지하고, 별도 실험은 같은 W0·FP32·binary의 batch 폭만 바꾼다.

사용자가 외부 LC0 비교 대신 내부 A/B를 선택했다. 고유 시작 이력 24개·흑백 교대 48판,
T2 100ms·128회 상한·고정 표본을 [로컬 강도 기록](research/LOCAL-STRENGTH-AB.md)에 잠갔다.
PGN 감사기는 기존 A Rules를 재사용하며 runner의 자동 claim 수락을 현재 Available
근거와 정확한 종료 문구로 검증한다. 기존 native V1의 explicit_claim 동작을 유지한다.
48판·24pair를 완료하고 A 자체 코어의 전체 PGN 감사와 독립 oracle의 이동열·최종 상태
대조를 통과했다. 폭 4 관점 20승·8무·20패, 득점률 50.00%, pair bootstrap 95% 구간
36.46~63.54%다. 양쪽 알고리즘은 같은 PUCT이며 동시 평가·batch 폭만 바꾼 실험이다.
강도 우위나 실제 처리량 개선은 확인하지 못했다. 폭 4의 WorkerLimit/LegalFallback
3건을 포함해 모든 판을 보존했다. 원본·정리본과 24pair PGN을 외부 reports로 회수하며,
정리본도 A 감사 결과가 원본과 같다. 매 착수 GPU 물리 drain·시계 공정성의 정식 인수와
per-game NN 완료 journal은 미검증으로 보존한다.

## 로컬 LC0·T1 후보의 실행·변환 조사

2026-10-04 후속 사용자 요청에 따라 공식 LC0 v0.32.1 Windows CUDA package와
T1 distilled의 원본 identity·제작자 직접 허가를 확인했다. RTX 4050에서 명시적
CUDA/FP16로 Maia/T1 각각 18회 UCI 착수·exit 0을 확인하고, 독립 oracle로 36개
입력·착수·PV와 흑/백 mate-in-one 12개를 감사했다. T1 FP32 ONNX export/checker와
LC0 Eigen 원본 ↔ ONNX CPU의 6개 상태·batch 2/4/8/16 수치 대조도 통과했다.
설정·원본/변환 digest·오차·공백과 B~F 순서는 [모델 전환 계획](research/LOCAL-MODEL-BASELINE.md)에 둔다.

이번에는 문서 계획과 **외부 도구의 실행/CPU 수치**를 인수했다. Rust의 T1 loader/
encoder·ONNX CUDA parity·실제 fine-tuning·새 정식 대국은 미실행이다. Maia exact
profile·공통 계약 0.1과 기존 48판의 `strength_eligible=false`를 유지한다. 기존 F의
합성 linear fixture를 실제 모델 trainer로 표현하지 않는다. 문서 경로는 Workspace CPU
workflow의 path filter에 포함되지 않아 이번 문서 PR의 CI는 자동 실행되지 않았다.

## BT4-it332 Rust 연결과 같은 가중치 LC0 벤치마크

2026-10-04 사용자가 BT4 적용·native LC0와 RoveZero 벤치마크를 지정했다. 공통 계약
0.1·자체 Rules/PUCT·Maia exact profile은 유지하고 C의 BT4 단일 source/export/profile,
C/D의 3 GiB admission 선언, B native 유한 1..4096 simulation 선택을 연결했다.
자연 worker 완료는 runtime shutdown 뒤 게시하도록 수정했으며 강제 마감·stop의
물리 GPU 시간 공정성은 별도다. BT4의 개별 weights license는 미확인·재배포 false다.

| 실제 인수 | 소스·관측 | 제한 |
|---|---|---|
| BT4 외부 CPU 참조 | LC0 `fd71a2d` 원본 Eigen ↔ FP32/opset17 ONNX, 12개 상태·batch 2/4/8/16 통과 | frozen 수치 대조, 학습/강도 아님 |
| Rust C/A/D CPU·CUDA | `aa0cc0247b9d7041e4525b2021363d0617cfd0bc`, 각 raw 12개·batch 1/2/4/8/16, 자체 A 상태 12개·history fill No 5/Repeat 7, drain/잔여 예약 0 | 그 뒤 변경은 UCI 네 파일이며 Cargo·A/C/D/encoding/contracts 영향 경로 동일성 확인에 한정해 수치 결과 재사용 |
| CUDA 실제 연산 | FP32·TF32 off·CPU fallback 없음, raw warm CUDA node 687·19파일 runtime bundle, 장치 전체 관측 최대 1,131 MiB | 개별 allocation peak/전체 VRAM hard cap/모든 race 아님 |
| 실제 B UCI 위치 query | `78b7c53502cadaff77fc6de5f0832eee55b9938c`, immutable binary `1a4b976b4110d251e21b09ed1e1f29b398b4ed01f533788a941de8dc4be969b3`, 18개 합법 bestmove·exit 0·confirmed final drain | D Computed 2,920·B root 초기화 18·non-root 소비는 process 집계, per-root physical journal 아님 |
| native LC0 비교 | 같은 BT4 원본, v0.32.1 Windows CUDA FP32/FP16 각 18 query·36개 printed PV 합법·exit 0 | FP16 별도 profile, Windows/WSL·CPU·batch 배치 차이를 순수 알고리즘 차이로 하지 않음 |
| 위치 benchmark | 총 54 query, LC0 FP32/RZ FP32/LC0 FP16 전체 장치 관측 최대 2,009/1,131/1,083 MiB | 250ms sampling·조건별 한 표본·RZ UCI NPS/PV 미제공 |
| 전술/마감의 후속 대상 | 즉시 메이트 LC0 각 6/6·RZ 0/6, RZ 일반 상태 5초 응답 최대 5,295.06ms·Expired/Stale 진단 보존 | 현 root는 방문 수 우선, 가치 부호 오류/BT4 인코딩 실패 원인 확정 아님; formal strength NO-GO 유지 |
| 검사/CI | 전체 workspace all-target/all-feature tests·strict Clippy·release build 통과, [37208852788](https://github.com/daejunnom/RoveZero/actions/runs/37208852788)의 두 OS SUCCESS 직접 확인 | CI와 실제 로컬 GPU/arena는 별도 증거 |

최초 native batch 16/min 1의 CUDA 오류와 batch 256/min 4의 후속 성공을 함께 보존한다.
ONNX·원본 hash·오차·옵션·cold startup·위치 benchmark와 후속 실행 지시서는
[로컬 모델 기록](research/LOCAL-MODEL-BASELINE.md)에 묶었다. 원시 자료는 저장소 밖
`reports/coordinator-integration/bt4-benchmark-20261004/`에 회수한다. native receipt를
처음 회수할 때 잘못된 directory glob으로 supervisor만 복사된 것을 확인하고 실제
`native-process-*`의 startup/termination을 추가 회수해 byte digest를 대조했다.
이 회수 보완은 실행 결과를 다시 만들거나 성공으로 바꾸지 않는다.

개발 대국은 4쌍/8판·같은 BT4 FP32·완전한 기존 opening history·500ms/수·별도
100ms host/transport 여유·256 total ply·전체 1,200초로 먼저 잠가 실행한다. 결과·
PGN audit는 종료 후 추가한다. 개발 pool·미확정 물리 시계/동일 자원 때문에 정식
강도 승격·Elo·실제 학습 인수는 하지 않는다.
