# 로컬 D02 native source-clock 계측 인수

기준일: 2026-10-03. 총괄 TASK-I02가 native 요청의 원래 시각과 실제 소비를 연결했다.
실행 소스는 `88ea26e714a56af81f43cd1180c31df08fe813e9`다. 후속 문서 head와 구별한다.
공통 계약 revision은 `0.1`이며 [PR #17](https://github.com/daejunnom/RoveZero/pull/17)에서 공유한다.
[이전 연결 인수](../INTEGRATION-STATUS.md#최적화-소스-b5ba853의-로컬-cpucuda-연결-인수)와
[의미 보존 최적화](PRE-RUNPOD-OPTIMIZATION.md)는 각각의 실행 근거로 유지한다.
이 기록은 D02의 요청 시각·실제 완료·소비를 맡는다. CPU fork probe나 외부 GPU 계획과
다른 연구 질문이므로 별도 문서에 두며 원본 journal·로그는 저장소 밖에 보존한다.

## 연결한 소유 경계와 비용

native CLI의 명시적 `--profile`로만 `SourceJournal<CompletionContext>`를 만든다.
8192개 metadata record와 출력 JSON 8MiB 한도를 고정한다. model·position·tensor·device
buffer를 journal에 보관하지 않는다. 기록할 때 직렬화·정렬하지 않으며 prefix와 누락 수를
보존한다. observer storage loss/poison은 평가·backup·취소·buffer 해제 권한을 만들지 않는다.
공유 journal의 lock·시각 취득에는 비용이 있으므로 계측 자체를 무료로 가정하지 않는다.

| 소유자 | 실제 계측 위치 | 의미 |
|---|---|---|
| B | selection 시작부터 checked request 제출 직전 | Rules 이동·input key·요청 준비의 CPU 비용 |
| C | projection prepare, native worker, 입력 staging, synchronous ORT Run, raw 출력 추출 | 인코딩·host 실행 구간·실제 물리 완료와 이후 변환 성공을 분리 |
| D | admission·dispatch·ready·validation·logical finish·delivery의 원래 owner tick | 새 clock origin을 만들지 않고 기존 ProcessClock의 Instant로 투영 |
| B | 기존 최종 tree guard가 들어 있는 accept 함수 | backup 확정 성공 여부; root 초기화도 별도 기존 aggregate와 대조 |
| UCI | 실제 stdout write/flush | process stream의 출력 비용; 개별 평가의 출력으로 배분하지 않음 |

CPU/CUDA V1 startup/termination의 JSON key/type topology를 이전 실행과 직접 대조했다.
기존 source error·quarantine·마감·generation·물리 pin·worker join을 유지한다. 공통 타입에
필드를 더하지 않았고 telemetry는 generic metadata만 받는다. 총괄은 B/C의 hook과 D의
original origin·execution ID·최종 guard·UCI 출력 수명 접점을 수동으로 맞췄다.

worker가 dispatch 함수 반환 전에 시작할 수 있으므로 그 두 사건을 강제 직렬화하지 않는다.
사후 검사에서는 요청 identity, 필수 사건, native 구간의 worker 내부 포함, 물리 완료 이후
ready→validation→delivery→backup 순서를 확인한다. 기록 순서와 사건 시각을 혼동하지 않는다.
누락·중복 stage·identity 불일치·overflow·unconfirmed shutdown을 complete로 표시하지 않는다.
ORT Run 완료 뒤 출력 변환이 실패하면 physical completion과 미소비를 함께 남긴다.

## 실행 명세와 실제 결과

로컬 RTX 4050 Laptop GPU 6GB·driver 610.62, Linux WSL2, Rust 1.96.0 release를 사용했다.
W0는 기존 Maia 1900 ONNX export, S0는 기존 최소 PUCT다. FP32, batch 1, native worker 1,
ORT intra-thread 1을 유지했다. 동일한 startpos와 `startpos moves e2e4`를 차례로 사용하고
각 root에 `go movetime 20000 nodes 128`을 주었다. NN root 초기화는 방문 수와 구분한다.
각 provider의 profile off/on을 순차 실행했으며 비교는 동작·소비 수의 대조다.

CUDA placement warm-up은 backend 생성 때 별도로 끝난다. journal은 그 이후의 **탐색 요청**을
기록하므로 전체 process의 모든 NN 호출이나 startup warm-up journal로 표현하지 않는다.
first timed invocation도 cold GPU 커널로 해석하지 않는다. 원본과 분석에 첫 표본 및 이후
257개 표본을 별도로 보존했다. batch 분포는 이 고정 B1 실행에서만 인수했다.

| 같은 실행 소스의 새 gate | 전체 gate 시간 | 합법 착수 | D computed / B root·non-root | 결과 |
|---|---:|---|---|---|
| CPU, profile off | 0.614초 | e2e4, c7c5 | 258 / B 수명 검사는 CPU V1 범위 | passed |
| CPU, profile on | 0.856초 | e2e4, c7c5 | 258 / journal 소비 258 | passed |
| CUDA, profile off | 14.490초 | e2e4, c7c5 | 258 / 2·256 | passed |
| CUDA, profile on | 24.056초 | e2e4, c7c5 | 258 / 2·256 | passed |

profile on의 CPU/CUDA는 각각 journal 3896개, 물리 실행 시도·완료·전달·guarded 소비
258개다. completed-not-consumed·physical-unconfirmed·worker failure·delivery rejection은 0이다.
storage/upstream loss·invalid interval·duplicate stage·identity mismatch·causal error·unbound
scheduler event는 모두 0이며 `journal_complete=true`, `accepted_request_timeline_complete=true`다.
원본 record의 원래 시각·context·execution을 별도 감독 분석으로 대조하고 quantile을 다시 계산했다.

각 gate에는 wall 120초, startup 45초, move 25초, quit 15초, cgroup RAM 4GiB·swap 0·
CPU 2 core·pids 128·파일/로그 상한을 적용했다. 모두 exit 0·confirmed drain·남은 owned PID 0,
원래 service/collection/mapping 오류 없음, OOM/kill 0·전용 cgroup 제거를 확인했다.
CUDA placement trace는 각각 CUDA Node 98·CPU Node 0이며 최종 mapped-image 감사도 통과했다.

전체 gate는 native bundle 복사·hash·model load·placement warm-up·protocol·종료·보고서 출력을
포함한다. CUDA off/on의 startup client 관측만 각각 13.260/22.298초였다. 이를 profile overhead나
추론 속도 차이로 옮기지 않는다. client의 go→bestmove는 off 0.300/0.310초, on 0.380/0.430초다.
그 시각은 외부 pipe 관측이며 engine의 내부 source clock과 합쳐 새 latency를 만들지 않는다.
한 번씩의 off/on 및 아래 258개 요청 표본은 통제된 D03 성능 개선의 판정이 아니다.

## 요청별 source-clock 분포

다음은 최종 CUDA profile on 실행의 nearest-rank P50/P95/P99다. 단위는 ms다.
단계 구간은 중첩될 수 있으며 분위수들을 더해 전체 지연을 만들지 않는다.

| 실제 구간, 각 258개 표본 | P50 | P95 | P99 |
|---|---:|---:|---:|
| B 준비 시작→최종 guarded backup 종료 | 3.371108 | 4.621393 | 5.615506 |
| synchronous ORT Run host interval | 2.349358 | 3.751160 | 5.041684 |
| C 물리 worker 전체 | 2.366105 | 3.790036 | 5.058361 |
| worker 완료→D ready 관측 | 0.676907 | 1.088653 | 1.197836 |
| B Rules/input/request 준비 | 0.071569 | 0.127679 | 0.197616 |
| C encoding preparation | 0.020349 | 0.026744 | 0.035904 |
| D admission→dispatch 시작 | 0.002483 | 0.044058 | 0.079477 |
| D validation | 0.000610 | 0.000925 | 0.001459 |
| 기존 최종 guard를 포함한 B backup | 0.002018 | 0.002829 | 0.003666 |

CPU의 준비→backup P50/P95/P99는 1.190385/2.348863/2.461157ms, ORT Run은
0.899074/1.120511/1.235712ms다. 같은 소형 모델의 직렬 B1 실행 관측이다. 장비 일반 성능이나
큰 batch·새 모델·LC0 강도에 대한 비교로 확장하지 않는다. 초기 `54b8fa5` pilot의 수치도 원본에
보존하며 최종 소스 표본과 섞지 않는다.

## VRAM·device timing과 D03 후보

실제 native PID의 45개 compute-app 질의에서 used_gpu_memory는 전부 `[N/A]`였다.
100ms 간격 whole-adapter의 관측 최대는 109MiB였으나 process/unsampled peak VRAM이 아니다.
CUDA gate cgroup peak 4GiB에는 native 파일 copy/cache가 포함된다. sampled process RSS는
약 0.94GB이며 이것도 VRAM이 아니다. 서로 다른 관측을 GPU memory peak로 대체하지 않는다.
[NVIDIA WSL 문서](https://docs.nvidia.com/cuda/wsl-user-guide/#features-not-yet-supported)도
일부 NVML 질의의 제한을 설명한다. 이번 환경에서는 실제 `[N/A]` 결과를 근거로 미측정 처리한다.

현재 placement JSON에는 Session/Node 사건만 있고 device Kernel 사건은 없다.
ORT Run host interval은 provider 내부 전송·실행·동기화를 포함하며 kernel-only/transfer 시간이 아니다.
[ORT profiling 문서](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html#gpu-profiling)의
CUPTI/`--enable_cuda_profiling` 조건을 만족한 별도 device trace를 확보하기 전까지 두 항목은
미측정이다. CUDA 도구가 있다는 이유로 해당 계측이 지원됐다고 가정하지 않는다.

동일 EvalInputKey의 추가 실행은 72개다. 각 root의 129개 입력은 내부에서 모두 고유하며,
중복은 **두 root 사이의 동일 game**에서만 나타났다. 이 fixture의 72/258을 실제 cache hit나
절약된 평가·대국 성과로 표시하지 않는다. D03의 첫 후보로 같은 game에서 root가 바뀔 때 쓰는
제한된 exact raw-evaluation cache를 검토할 수 있다. model/encoding/backend/precision/compute와
실제 이력 입력을 함께 식별하고 새 요청의 합법 수·generation·deadline·최종 guard는 다시 검증해야 한다.
`ucinewgame` 초기화와 cache의 메모리·capacity·eviction 예산도 먼저 고정한다.

기존 native V1과 이 journal의 현재 인수는 computed B1 경로다. cache hit를 물리 실행으로
속이지 않도록 별도의 provenance·receipt·source-timing 소비 경로를 연결한 뒤 baseline/variant를
비교한다. 캐시 조회를 새 방문으로 만들지 않으며 정상 selection의 소비만 한 번 backup한다.
ready 관측 지연도 다음 후보지만 cache와 polling/batch 변경을 동시에 넣지 않는다.
D02 host-source 인수는 통과했고 device transfer/kernel·process peak VRAM과 D03 통제된 GPU A/B,
실제 학습·정식 paired 강도·통계 인수는 남아 있다.

## 검사·보존·재현 식별

WSL Rust 1.96.0의 `cargo test --workspace --all-targets --all-features --locked --offline`은
65개 test group에서 **682 passed / 0 failed / 16 ignored**다. fmt·동일 범위 strict clippy도 통과했다.
ignored 검사를 전체 실제 GPU 인수로 승격하지 않는다. 같은 실행 소스의
[CI 37119707352](https://github.com/daejunnom/RoveZero/actions/runs/37119707352)는 Ubuntu·Windows
각 job이 실제 성공했다. checkout SHA·extended perft·독립 python-chess·F fixture·clippy step의
성공을 확인했다. 원래 shell wrapper의 exit 전달 오류와 성공한 Rust 로그를 보존하고, 최종
검사는 subprocess의 exit code를 별도로 기록해 모두 0임을 확인했다.

| 식별 항목 | SHA-256 |
|---|---|
| root Cargo.lock | `156aa1876ac435d28fbebebe88e27228b49398a8a822687845105a1d2e855c76` |
| 실행 파일, 3,135,264 bytes·single link·readonly | `ce591331426f290e0827d55e98a686f730c11bc19d0d2ac86098e64784288069` |
| 최종 유한 감독 recipe v2 | `009796f790f6c8e4b93ac454816103e5130112fcd62683dcd56712462ac70777` |
| 최종 분석 | `460840f2e88b64a5082d5526b52894a973ed3d7b37bb1996695b27c0bae63256` |
| 회수 inventory | `d0e08a35018d3da35f0852f37a35aa080417e5f25841003e2af7b64a80ff4ad2` |
| 회수 receipt v2 | `6dc195fdb8f12a7b849678a17b157a3f30320800468526e50c3cc355bf087031` |

원본 metadata/journal/log/placement/recipe 73개·11,917,030 bytes를 Windows 외부
`reports/coordinator-integration/local-d02-20261003/`에 회수하고 모든 bytes/hash를 대조했다.
archive는 495,074 bytes, SHA-256은
`787c395dd49d6b2cf87be730cf7b93d55e41f9433d7b6fba652556b4276e40b2`다.
첫 Windows 직접 압축 입력 해제 실패를 보존하고 원본 archive를 변경하지 않은 채 gzip을
올바르게 해제한 새 capture v2에서 검증했다. 원래 Linux 자료·초기 pilot도 유지한다.
native library·실행 파일·weights·cache·개인 경로는 Git 요약이나 회수 payload에 넣지 않았다.
상세 실행 명령·고정 asset/bundle hash·자원·출처는 외부 recipe/setup/receipt에 남긴다.

## PR #17·#18 병합 준비와 보고서 저장 보완

2026-10-04, 총괄 TASK-I02가 PR #17의 계측과 PR #18의 최적화를 수동 연결했다.
기존 GPU 인수는 위 실행 소스의 이력이며 이번 재검사는 **CPU·FP32**다.
PR #17 제품 수정 소스는 `b83012145b16c959c1df4121efaf4d00503b5162`, 두 PR 통합
재검사 소스는 `a93569bedb802a4eb07f19b12241595715d21df1`이다. 공통 계약 0.1과
8192 record·출력 8MiB 한도는 유지한다.

통합 전 첫 CPU 실행 `673f3e4`는 합법 착수 두 개와 물리 drain을 기록했지만, 종료
10초 한도를 넘겨 실패했다. Windows에 마운트한 출력 디렉터리에서 serde의 작은
write가 파일에 바로 전달되어 JSON 387,606 bytes가 불완전하게 남았다. 실패 로그와
부분 JSON을 보존했다. `ProfileWriter`는 producer 종료 후 **64KiB BufWriter**로
저장하며, byte cap 검사·명시적 flush·파일 sync와 실패 전달을 유지한다. 실제 저장
helper의 검사로 완전한 JSON·write 합치기·8MiB 초과·flush 실패를 확인했다.

통합 소스의 새 실행은 같은 CPU 모델로 두 root에 `go movetime 10000 nodes 32`를
보냈다. 합법 착수 `e2e4, c7c5`, exit 0·confirmed drain, 실제 물리 시도/완료/전달/소비
각 **66개**, 손실·불일치·미소비 0을 기록했다. JSON의 journal과 accepted timeline이
완전하며 quit 송신부터 exit까지 **약 0.064초**였다. 이는 유한 종료·저장 회귀 검사
한 회의 관측이며 통제된 성능 A/B나 GPU 결과가 아니다.

PR #17 수정 소스의 [CI 37158044723](https://github.com/daejunnom/RoveZero/actions/runs/37158044723)는
Ubuntu workspace **683 passed·16 ignored**, Windows **629 passed·2 ignored**이며
실패 0이다. fmt·기본 native CLI·release 독립 Rules 대조·Python model 도구·strict
Clippy도 두 OS에서 성공했다. ignored 검사는 미실행으로 유지한다. 통합 소스의
추가 소비자 검사와 원시 근거의 논리 경로는
[PR #18](https://github.com/daejunnom/RoveZero/pull/18)의 최적화 기록 12장에 기록한다.

통합 이후 D02 v1은 **기본 B1·fresh Computed** 실행만 지원한다. `--profile`과
raw cache·폭 2 이상·buffer 재사용·I/O Binding·CUDA Graph를 함께 요청하면 asset
로딩 전에 명확히 거부한다. C profiled worker와 backend에서도 같은 제한을 검사한다.
해당 실험에는 cache provenance와 batch의 물리 실행을 표현하는 별도 계측 인수가
필요하다. 기존 GPU profile을 새 옵션의 측정으로 재사용하지 않는다.
