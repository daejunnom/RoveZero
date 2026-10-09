# PALS 구현 현황과 후속 인수 지시서

이 문서는 구현된 접점과 현재 실행 증거를 정리한다. PALS의 단계·V3 명세·비교 질문은
[기존 PALS 구현 계획](../PALS-IMPLEMENTATION.md)이 소유하며 여기서 새 계획으로 복제하지
않는다. 사용자가 제공한 `RoveZero_PALS_Internal_Algorithms_KO.md`와
`RoveZero_PALS_Architecture_Flows_KO.md`는 설계 자료다. 원문의 학습·실행 제안을 실제
승인으로 확대하지 않는다. 현재 목표는 **독자 Rust 엔진과 모델·학습 준비의 구현이며,
실제 optimizer 학습은 제외**한다. 생성 가중치는 미학습 초기화 자산이고 기력 인수는 없다.

Rust 제품의 실제 startup·역할 소비·취소·게임 수명·최종 증거 인수와 Python reference의
export·frozen forward·dataset preparation을 구분한다. Python 준비 성공을 Rust 제품의
실행 인수·학습·강도 결과로 승격하지 않는다.

## 2026-10-10 현재 소스 인수 상태

실제 optimizer 학습을 제외한 선택 구현·실행 인수는 완료됐다. 선택16행의 범위는
[PALS 구현과 인수 기록](../PALS-IMPLEMENTATION.md#2026-10-10-현재-소스-인수-상태)을 따른다.
이 절 뒤의 각 실패·보류·미완료 기록은 당시 source와 실행의 이력이다. 새 정상 실행으로
과거 원문을 고치지 않으며, 미관측 범위를 성공으로 채우지 않는다.

**소스·CI·등록 이미지.** 기존 제품 실행 source는
`935519d8e2cf3ddb3fe25d52da8497700d7cf1a5`, 최종 consumer source는
`cbaea3b8e92669adcdb4acefd1c6a71fe4d83524`다.
[소스 CI 37973912091](https://github.com/daejunnom/RoveZero/actions/runs/37973912091)의
Linux·Windows·CPU bindings·PALS model 네 job은 실제 성공했다. build536은 fmt,
captured CPU witness4, pair example7, workspace all-targets/all-features Clippy 및 release
build를 통과했고 등록 source265개를 빌드 전후와 실제 실행 감사에서 대조했다. 실행된
consumer는7,949,896bytes/SHA-256
`00884ce715e4353ee6c29f03fd0f2750ff887ae18cf403c1b29cb2c9c3933117`이다.
최종 문서 tip의 별도 CI·검토와 실행 결과 재사용 확인은 외부 영수증/PR에 남긴다.

**기존 source935의 실제 수집503·GPU508·대국514.** R5는 같은 continuation 안에서
3Reply를 실제 소비한 두 사례를 인수했다. 전체94개 role 완료·전달·소비, backend NN188개
(public94+private94), CPU8192nodes/32jobs를 각각 기록했다. strict frozen loader와
current-view·분할/누출 소비를 확인했고89개 record는 holdout/value masked다. 두 게임은
ply_limit으로 결과 unknown이며 관측되지 않은 WDL·ranking·utility target을 만들지 않았다.
기능 인수는 `reports/pals/r5-current-source-503-final-acceptance.json`에 연결한다.

GPU508은 source935/build492 제품의 beam4·정상 refine1이다. P13/C8, role21개의 물리
완료·전달·탐색 소비, search NN42개(public21+private21)와 startup3개를 구분했다.
CPU16task/521nodes, 준비13.306622초/go0.289842초, cgroup peak3,582,324,736bytes와
OOM0을 관측했다. Repair stage8은 Repair NN 실행과 다르며 accepted Repair NN과
supported Repair는0이다. reset ACK2회/generation3·최종 빈 bank·in-flight0·shutdown/
buffer 해제·EOF/reap/group/unit 종료를 인수했다. 독립 보고는
`runs/pals/gpu-product-full-refine-20261010-02/independent-observed-summary.json`,
6,810bytes/SHA-256 `49d8b91f21ad9c0b0b70ed979ca5409bd90158e8e218eeeec0409ba45a581e05`다.
과거 source2ad의4control/0round smoke는 영향 소스181개·이미지·빌드 조건 동일성으로
source935에 재사용 확인했다. 이는508의 실제 정상1round와 다른 실행이다.

R6 source935의 `pals-current-own-cpu-r-935-03`은 자체 CPU_R 상대의 흑백 교환2판/99ply,
120초+1초, 표준 시작, process restart의 기능 pilot이다. 백 PALS76ply0-1과 흑 PALS23ply1-0은
모두 Rules checkmate였으며 scored2/incomplete0·네 endpoint의 typed failed-go0을 확인했다.
실제99개 raw clock의 before 연속성·ceil 벽시계 차감·적시 착수의1000ms increment와
position/history/go/bestmove↔PGN 수순을 대조했고 오류0이었다. PGN의 반올림된 시간 주석만으로
시계를 인수하지 않았다.

NN4,620개(public2,310+private2,310)와 role 결과 소비2,310개는 다른 단위다. PALS 자체
CPU_R 요청1,282/새 완료1,215/재사용 완료 소비2/완료 소비1,217, nodes633,189를 기록했다.
partial iteration55/frontier 소비12/cache 값 소비279를 요청 깊이 완료나 새 작업에 더하지
않았다. 두 PALS process 각각의 public/shared_pc ORT session2개와 양쪽 물리 shutdown,
buffer 해제·in-flight0·quarantine false를 확인했다. root54.265초·exit0·reap·EOF·unit 종료,
원래900초+정리30초를 인수했다. 전역 affinity0/2·memory.high6GiB/max12GiB/swap0는 관측했고,
전체 memory peak·개별 endpoint peak·VRAM은 unknown이다. PGN은
`runs/pals/pals-current-own-cpu-r-935-03/paired-recovered/paired-attempt-01/match.pgn`,
5,647bytes/SHA-256 `b86d36a9cf248e51482c5df6bfc422679a9db26de2502d2488dcf491c8ca9826`이다.
최종 기능 인수 `reports/pals/r6-final-acceptance-514-KO.json`은8,080bytes/SHA-256
`1a57739c6d7e8643be036a19f1796d6e684c1cab45b57c2579e88d8817148b63`다. 미학습 PALS0/2를
기력·Elo 판정으로 승격하지 않는다.

**R2/R3 실제 Repair·독립 witness·다음 Query537.** ONE Query의 두 action 준비531은
실제 Query factory1회·prepared2회·caller binding2회를 기록했다. 사전 Query는
`7bccb44d065167c3df91ae2d47beb1d7d338afe53869aad5c6c4f68ca33c270a`다. 후속537은 이 입력과
등록 모델10/CPU FP32/Fresh, N4096·4N source·8N 전체 예산을 사용했다. frozen source4-stage는
4 CPU task/278nodes·NN28을 보고했고 독립 CPU4-stage/278nodes·캡처14입력 NN28의 실제
완료를 확인했다. source child의 물리 작업 attestation은 false이고 source scalar 점수 일치는
unknown/null이다. reported/source와 fresh independent scope를 합친 비용임을 명시한다.

next Query는 `75ac23d290f0b7a978f2837a97ce06a019443b7584d0c0f2b8c040b30f777735`, prior ledger는
`5c19de62204aa35da52001bd53e0da62329b7feb9945e44b90e311820109caa6`에서
`bf118682e1525f2d1a6f6b15b4f0d0ec9640256fa38636ae9c546cc3e02d40dc`로 전이했다. 새 Query는
source-bound evidence를 실제 소유하며 JSON capability 복원·Query2 action 의미의 재검증·
product verifier 권한을 주장하지 않는다. 다음 coverage 선택은 Defer, 추가 V/NN/CPU0이다.
원래 S→다음 선택 완료의 전체 비용1.539106541초/CPU556nodes/NN56, root2.047초/exit0·reap·
EOF·unit 종료, 독립 physical in-flight0·shutdown/buffer 해제를 인수했다. 회수16파일/2,550,025bytes는
모두 inventory hash와 일치했다. 원시 근거는 `reports/pals/r2-basic4n-537-recovered/`와
`r2-basic4n-537-outer.json`이다.

**R4 동일 process의 두 action538.** 같은 사전 Query/prior의 N4096/N2048을 nested typed
owner로 묶되 각 original S/W와 독립 model owner를 보존했다. 각 W300초/정리 reserve30초,
pair670초=2W+준비30초+비교10초+정리30초, root727초를 고정했다. 두 lane의 독립 witness는
서로 다른 epoch/lease를 사용했고 각각14개 캡처 입력/NN28 완료와 in-flight0·shutdown/buffer
해제를 확인했다. 양쪽 child exit0/group gone, root3.641초/exit0·reap·EOF·unit 종료/정리 오류0이다.

왼쪽 전체 비용1.605115383초, 오른쪽1.325331493초이며 각 CPU556nodes/NN56이다.
합CPU1112≤49152·NN112=한도112를 지켰다. 서로 다른 예산 상한이 실제 작업량 감소를 만든
것으로 해석하지 않는다. 원래 백 관점에서 repair0/counter+16이라 조건부 counter-lower가
성립하지 않았다. production의 `mask=preference.is_some()`에 따라 preference null/mask false,
actual utility groups0·utility/target/training 권한 false를 보존했다. 시간 차이로 이 조건 실패를
덮지 않았으며 positive Pareto·paper reward·learned V·기력·속도 향상은 주장하지 않는다.
이는 실제 음성 비교·unknown 처리 인수다. source child scalar 일치와 physical attestation의
한계도 그대로다.

회수36파일/5,107,001bytes의 inventory와 source265개를 독립 대조했고 blocker0이었다.
`reports/pals/final-audit-538.json`은18,897bytes/SHA-256
`4dfee8b9de37df15ff1c7c2076654a620b5e450cfd229621ea4d259fa629d957`, 상태는
`PASS_MASKED_OBSERVATION`이다. 원시 pair acceptance/conditional cost와 양쪽 witness/closure는
`reports/pals/r4-pair-538-recovered/`에 보존했다.

**수정·재사용·보존.** 532의 full4-stage 부분 성공 뒤 independent witness가 거부된 원인은
producer의 full OWNED descriptor와 consumer의 bare search_conditions 대조 오류였다.
consumer에 실제 descriptor 전체 조건을 연결하고 positive 및6축 tamper 검사로 보완했다.
조건·완료·깊이·예산 검사를 약화하지 않았다. 935→cbaea3b의 변경은 pair example, 추가 CPU
descriptor 메서드, 독립 witness의 세 파일이다. producer engine/sourceExpected7·모델·Rules·
runtime·제품 UCI 경로·Cargo·workflow는 유지됐다. source935의 수집/GPU/대국을 cbaea3b의
새 물리 실행으로 표시하지 않는다. selected-source 재사용 근거
`reports/pals/final-selected-reuse-539.json`은4,843bytes/SHA-256
`ec930882c0b548912d092119f1ecfe39d545dd0902465df5b8c5c86ff6b33b5b`다.

524~527의 cwd/profile/store/runtime-root pre-NN 거부, 528의 prefix 불일치·Repair not ready,
532의 consumer 거부와534/535의 검사 실패는 원문 그대로다. 534의 잘못된 test filter0개를
witness4개 PASS로 세지 않았다. 진단 뒤 고른 b2b4/b8a6 사례530/531은 기존 result-free
등록517/522와 구분하며 넓은 성능·기력 일반화의 근거로 쓰지 않는다. cleanup529는 Linux와
Windows의 비활성 private 사본을 각각6개/122,609,586bytes, 합245,219,172bytes 논리 반환하고
원본 공유 자산·CAS·PGN·로그·negative 보고서를 보존했다. VHD 물리 축소 주장은 없다.

선택된 비학습 구현·정상 실행·음성 비교는 인수했으며 실제 optimizer 학습은0이다.
GPU508의 Repair NN0, 미실행 full native quarantine fault·개별 GPU late-completion trace,
unknown VRAM peak·물리 parameter sharing·source scalar 일치를 보존한다. CUDA private
Warm·추가 pruning·대체 optimizer·실제 학습과 대규모 강도 평가는 후속 범위다. 미학습 모델의
준비·기능 인수를 학습 완료·경쟁력·성능 개선으로 바꾸지 않는다.

## 2026-10-09 병렬 재개 감사 기준

이 절의 기준 소스는 `b6de5325d47a417461823f4a92329287c0e5af09`다. 사용자는 학습
제외 목표의 병렬 구현을 재개했다. GPU 보류와 CPU 실행 보류를 서술하는 뒤의 기록은
그 당시 상태이며 현재 재개를 취소하지 않는다. 총괄이 최종 소스·자산·환경·예산을
확인한 실행과 개별 담당의 source 작업을 구분한다. 기존 실패·unknown·미인수 원문과
실행 자료를 변경하거나 새 성공으로 재작성하지 않는다.

기준 SHA의 [CPU CI 37897124887](https://github.com/daejunnom/RoveZero/actions/runs/37897124887)는
Linux·Windows·CPU bindings·PALS model 네 job이 모두 성공했다. 이것은 후속 통합 소스의
CI, 실제 registered frozen 4N 실행·next Query·whole-action utility·최종 다중 Reply 수집,
GPU 또는 paired 인수의 성공을 뜻하지 않는다. 원문의 전체 요구사항 대조와 최종 실행
순서는 [PALS 구현과 인수 기록](../PALS-IMPLEMENTATION.md)을 따른다.

| 재개 시 경계 | 확인된 소스·기존 증거 | 최종 통합에서 필요한 증거 |
|---|---|---|
| CPU_T SEE | private Rust CPU task와 producer의 명시적 SEE 선택까지 구현돼 있다. 기준 SHA의 Linux·Windows CI는 `legal_see_is_explicit_and_preserves_legacy_identity_and_resume_boundary`, `see_profile_conditions_and_reports_use_actual_selected_engine`, `see_cross_profile_keeps_ordering_but_uses_fresh_independent_namespace`를 통과했다. | 실제 선택한 producer 실행은 source·binary·ordering 조건으로 별도 등록한다. 기본값 변경·기력 개선을 이 검사로 주장하지 않는다. |
| 실제 frozen caller와 Query 환류 | loaded binary·원래 실행 창·보고 Rules·caller 비용·단일 prior 자료 발급의 소스와 CPU 검사가 있다. | 실제 registered child/NN·독립 native witness와 다음 Query admission·episode ledger·중복/늦은 결과 거부를 연결한다. 발급 자료를 legacy receipt 또는 native capability로 바꾸어 쓰지 않는다. |
| V 비용·utility | 조건부 witness와 엄격한 unknown/masked 준비 경계가 있다. | 같은 사전 질문·prior의 독립 action 전체 비용과 실제 결과가 있어야 알려진 utility를 발급한다. 부분 관측이나 선언값을 양의 target으로 승격하지 않는다. |
| GPU record page와 물리 수명 | 명시적 resident packing의 loader·planner·owner·CLI·arena 소스와 `34149e2`의 실제 numeric/resident·새 게임 초기화·정상 worker 종료 자료가 있다. 과거 numeric·제품 오류 자료도 각 실행 source로 보존한다. | 제품 UCI stop·취소·늦은 root·미확정 완료 인수는 남는다. 새 source에 대한 재사용은 영향 소스·자산·환경을 대조한다. host page·CPU fixture나 checker 정상 종료를 제품 실패 경로의 대체물로 쓰지 않는다. |
| 다중 Reply와 최종 CPU pair | actual C continuation·anchor·native journal과 typed failed-go·Rules/시계·최종 저장 경로가 있다. 기존 CPU03 및 더 이전 pair의 실패 자료를 보존한다. | 같은 최종 source/binary의 실제 collection 및 자체 CPU_R paired 한 쌍에서 모든 계수·영수증·PGN·EOF/reap/group 종료·정리를 확인한다. 과거 build89 준비는 현재 source의 실행 증거가 아니다. |

## 실제 GPU 수치·resident 인수와 R7 잔여

2026-10-09 `gpu-numeric-resident-20261009-01`의 실제 자료를 직접 읽고, summary가 지정한
네 파일의 bytes/SHA를 원본에 대조했다. source는
`34149e2cc6e588cf32452aece0494aeb8cb05134`, release checker `pals_model_check`는
3,286,384 bytes와 SHA-256
`46c8eb1b522c45f1e0e46f34ba88d98c4c9898c31afee503b4c4759477c8bd1c`다. checker·runtime·
source/asset 관측은 실행 전후 일치하고, launcher의 이후 source HEAD도 같은 SHA다.
후속 변경이나 최종 PR HEAD를 이 실행 SHA로 대신하지 않는다.

| 실제 검사 범위 | 관측과 인수 |
|---|---|
| 실행 구성 | WSL Ubuntu의 CUDAExecutionProvider, FP32, TF32 off, 미학습 P/C export, 독립 reference와 registered packing. 실제 학습은 false다. graph placement/profile의 CPU 제어·전송 허용과 CUDA neural 실행 근거를 함께 보존한다. |
| 독립 수치 | reference 6개와 resident anchor 6개를 대조했다. resident derived 15개와 repeat 15개는 record append·correction·순서/ID 이동·eviction·역할 순서·빈 padding 등 선언한 파생 경계를 검사했다. 이것은 실제 Rules 판정·전략적 강도 인수가 아니다. |
| resident raw 차이 | 후보 logit 최대 절대 `2.682209014892578e-7`, policy `4.470348358154297e-8`, WDL logit `2.384185791015625e-7`, WDL `5.960464477539063e-8`, latent `1.430511474609375e-6`. logits 절대 `1e-4`·상대 `1e-3`, policy/WDL 절대 `1e-4`의 기존 오차를 만족했다. |
| 실제 NN 계수 | resident public 완료 10·private 완료 37·합계 47. anchor/derived/repeat의 개수는 case scope이며 물리 NN 입력 수와 다르다. whole comparator의 완료 30, 기존 main numeric backend의 완료 20은 별도 counter scope다. 이들을 47에 더해 제품 처리량·탐색 소비량으로 표시하지 않는다. |
| 새 게임 초기화 | `game_generation=2`, live block·certified projection·whole-owner host/device bytes 모두 0. worker metadata ACK·physical shutdown과 legacy backend의 physical completion/shutdown이 confirmed다. |
| 실제 감독 종료 | checker exit 0·PID reap·normal exit·process group empty, outer returncode 0·unit inactive/dead·ControlGroup empty, timed_out=false. 이는 해당 checker 실행의 관측이다. |
| 시간·자원 | checker 38.095645초, outer 43.172초, resident 부분 4.268912초. CPU affinity `0 2`, high 6GiB/max 12GiB/swap 0. 새 cgroup peak 3,644,235,776 bytes, memory.events의 low/high/max/oom/oom_kill/oom_group_kill 모두 0. Windows 실행 전 커밋 여유 11,358,842,880 bytes는 admission 시점 관측이다. |
| 저장소 정리 | managed cleanup exit 0, temporary_removed·tree_cleanup_verified=true. 선언 cap·slot 수와 원래 실행·보존 조건을 유지한다. 이것은 다른 보존 모델·run·보고서의 삭제나 전역 저장소 공간 반환 증거가 아니다. |

원본 증거는 관리 루트의 `runs/pals/gpu-numeric-resident-20261009-01/`에 보존한다.
공유 문서에는 개인 절대 경로·계정·GPU 호스트를 넣지 않는다. 확인한 pin은 다음과 같다.

| 증거 파일 | bytes | SHA-256 |
|---|---:|---|
| `launch.json` | 3,625 | `bd4e61caa7a68d2eeb6f93cf0843e7f3718f56fe7189845801ee187e01290a3c` |
| `execution.json` | 24,000 | `f4a11ee1b05cfac7da5f9b552162bba63d2bf33043c885f08d643d3b7e2b619f` |
| `launcher-execution.json` | 341 | `30b5a98c63a052a99feeab5002cf4bcea4393448fda525de33ba4771861356f5` |
| `numeric.json` | 309,212 | `80074b97f066b016377205f3d0540beee028fd78658e70dcaae71f9346bf5076` |

VRAM peak·네이티브 파라미터 실제 저장 공유는 `unknown`이다. device-backed public K/V의
host tensor 대조는 복사하지 않아 미실행이다. 선언된 resident 자원 한도와 관측한 cgroup
peak를 혼동하지 않으며, Windows 전체 커밋 peak도 이번 관측에 없다. 전체 실행은
독립 checker 및 exclusive physical worker의 수치·정상 종료 인수다. 제품 UCI stop/cancel·
새 root의 늦은 결과·미확정 완료/격리 경로는 numeric checker에서 실행하지 않았다.
기력·속도·메모리 개선, 실제 학습, R2~R6의 실행 완료를 주장하지 않는다.

### 기록 입력 재추론 witness의 용어와 권한

새 source-owned 경계의 assurance scope는
`captured_nn_input_reinference_and_independent_cpu_condition_reexecution`이다.
[`captured_repair_witness.rs`](../../crates/rz-arena/src/pals_replay/captured_repair_witness.rs)는
원래 capture/material owner가 소유한 P/C 입력과 raw 출력 bit를 다시 대조하고,
[`replay_captured_cpu_witness.rs`](../../crates/rz-uci/src/pals_cpu_task/strategic_action/replay_captured_cpu_witness.rs)는
조건·Rules snapshot을 고정한 새로운 자체 CPU 실행에서 실제 endpoint fact를 얻는다.
`CheckedCapturedRepairWitness`가 인증하는 범위는 **이 새로운 실행**이다. 원래 child의
과거 물리 NN 실행이나 권한을 같은 관측으로 자동 인증하지 않는다.

기록된 `query[7]`은 당시 deadline feature이며 **historical input**으로 byte/bit와 입력
fingerprint를 유지한다. 현재 남은 시간으로 재작성하지 않고, 새 live Query 시계로도
사용하지 않는다. 재추론·독립 CPU의 실제 준비/완료 시각·취소·비용은 별도로 관측하고
원래 absolute S/E/W에 결합한다. 과거 feature가 보존된다는 이유로 새 예산을 발급하거나
원래 deadline을 다시 시작하지 않는다.

원래 CPU scalar score는 구조화된 report에 없으므로
`source_child_score_equivalence=unknown_not_structurally_reported`이며 audit의
`source_cpu_scalar_score_equal`은 null이다. 새 독립 CPU의 실제 score/조건/완료/노드는
별도 사실이고, source child work는 reported scope로 남긴다. NN raw bit 재추론 일치와
원래 CPU score 동등성을 하나의 사실로 합치지 않는다. reported score text를 파싱하거나
새 score를 과거 score로 채우지 않는다. 이 경계의 source·CPU fixture와 실제 R2 실행,
R3 Query admission·R4 whole-action utility 권한은 각각 인수한다.

### 남은 최종 인수

| 항목 | 현재 남은 실제 증거 |
|---|---|
| 제품 GPU 수명 | registered UCI/backend의 stop·cancel·늦은 result/새 root·미확정 완료와 정상 drain/종료. 독립 checker의 정상 shutdown으로 대체하지 않는다. |
| R2 | 등록된 frozen 4N capture, 기록 입력 재추론·독립 CPU 조건 실행, loaded binary·원래 S/E/W·exact raw·Rules·최종 물리 shutdown·저장. 원래 score의 unknown은 유지한다. |
| R3 | 다음 immutable Query의 실제 prior 소비·입력 seal·episode/ledger 순서와 중복·늦은 응답 거부. |
| R4 | 같은 사전 조건의 독립 action들에 대한 실제 전체 비용·결과·불확실성과 whole-action utility. 부분/unknown에서 label을 발급하지 않는다. |
| R5 | current-source actual 다중 Reply tensor·dispatch·소비·Repair anchor/revision·수집·consumer·split/leak 검사와 종료·저장. |
| R6 | current-source/binary·actual C continuation S lane·동일 모델/예산/시계로 자체 CPU_R 흑백 교환 2판. 준비 child actual exit admission·NN 소비·typed failed-go·Rules/clock/PGN·process/storage·필수 회수. |
| R7 | 원문 전체의 선택된 필수 요구사항을 최종 source/feature·CPU CI·actual CPU/GPU/paired 증거에 대조하고 final SHA의 CI·PR 인수 기록을 확인한다. |

이 절 뒤의 GPU 보류·미실행·실패 문구는 그 당시 source/실행의 이력이다. 새 정상 검사로
지우거나 소급 승격하지 않는다. 실제 학습 제외, V-free 제품, 기력 인수 부재와 원문의
선택적 후속 연구 경계는 유지한다.

## 처리 흐름과 소유 경계

제품 경로는 기존 UCI에서 탐색을 선택한 뒤 Rules의 정확한 상태·이력·합법 수를 사용한다.
`--search=puct`, `--search=cpu`, `--search=pals`는 별도 선택이며 PALS 작업을 기존 PUCT의
simulation·visit으로 해석하지 않는다. PALS는 P의 수순 제안 → 자체 CPU의 조건부 검사 →
C의 이탈 지점·반박 → P의 수선 → 관련 상황의 결론 갱신으로 진행한다. 마감 전 마지막
유효 착수를 선택하며, 모델의 추정은 Rules의 체크메이트·스테일메이트 판정을 대신하지 않는다.

| 소유자·소스 | 책임과 교체 경계 |
|---|---|
| Rules / [`rz-position`](../../crates/rz-position/src/lib.rs) | 합법 수, 특수 수, make/unmake, 완전 상태·이력·종료의 단일 소유자. |
| Own CPU / [`cpu.rs`](../../crates/rz-search/src/cpu.rs), [`cpu_value.rs`](../../crates/rz-search/src/cpu_value.rs) | CPU_T/CPU_R가 공유하는 유한 iterative deepening·PVS·aspiration·quiescence·TT와 평가 접점. profile·조건·depth·node·시간·완료 범위를 따로 식별한다. 초기 평가를 학습된 평가로 보고하지 않는다. |
| PALS Search / [`engine.rs`](../../crates/rz-search/src/pals/engine.rs) | 제안·반박·수선, 제한된 continuation 확장, CPU 작업 요청·소비와 루트 결정. `RoleModel`의 합법 후보 순서 logits·WDL 접점으로 모델을 시작 시 선택한다. |
| Canonical stores / [`store.rs`](../../crates/rz-search/src/pals/store.rs) | 상태·공유 line chunk·관측·상황·의존 관계·작업과 소비자별 완료를 보존한다. representation cache 회수와 검사 결과 삭제를 혼동하지 않는다. |
| PALS 계약 / [`pals.rs`](../../crates/rz-contracts/src/pals.rs) | `pals/0.1` 역할·authority·generation·situation handle·representation key·typed payload·CPU 조건. 총괄이 공통 계약을 소유한다. 기존 평가 계약 `0.1`의 가짜 `EvalOutput`으로 변환하지 않는다. |
| Runtime / [`rz-runtime/pals.rs`](../../crates/rz-runtime/src/pals.rs) | 큐·batch key·deadline·취소·물리 lease·typed 응답과 role-neutral/private memory namespace. 정상 물리 완료 전 입력·출력·session·workspace를 해제하거나 재사용하지 않는다. |
| 모델·ORT / [`pals_model.rs`](../../crates/rz-eval/src/pals_model.rs), [`pals_onnx.rs`](../../crates/rz-eval/src/pals_onnx.rs) | 입력·shape·dtype·유한값·예산·output 의미, 자산 pin, 공개 K/V 및 역할 graph, 하나의 물리 worker. cache hit·물리 NN 입력·탐색 소비를 별도로 기록한다. |
| 제품 조합 / [`pals_native.rs`](../../crates/rz-uci/src/pals_native.rs), [`main.rs`](../../crates/rz-uci/src/main.rs) | 정확한 Rules→모델 입력, 역할 요청→runtime, 모델 소비 ACK, UCI 시계·중단·새 게임·종료. 명시적 CPU/CUDA provider와 GPU startup 경계를 구현했다. CUDA-control 경로는 독립 pin의 typed inventory 및 실제 placement/profile witness를 요구하며 GPU 인수 상태는 아래 실행 기록과 구별한다. |
| 실행·데이터 / [`pals_manifest.rs`](../../crates/rz-experiments/src/pals_manifest.rs), [`pals_data.rs`](../../crates/rz-experiments/src/pals_data.rs), [`pals_collect.rs`](../../crates/rz-arena/src/pals_collect.rs) | V3 명세·lock·receipt, own-source 데이터·시점·입력 seal·분할·누출 검사, 수집 및 실패 보존. 외부 상대 엔진과 내부 모델 선택은 독립이다. |

CPU profile과 모델·입력의 의미가 달라지면 기존 작업 재개·cache 자격을 다시 검사한다.
CPU가 중단됐을 때 남은 completed depth와 frontier estimate를 구분하고, 미완료 결과를
요청 depth의 완료 bound로 저장하지 않는다. CPU와 WDL의 서로 다른 척도를 임의로 더하지
않는다. 외부 UCI 상대를 선택했다고 외부 엔진이 PALS 내부 CPU_R 구현으로 연결되는 것은
아니다. 실제 교체에는 capability·문제 조건·raw score·완료 범위의 별도 adapter가 필요하다.

## 원문 설계와 이번 구현의 범위

| 설계 항목 | 현재 구현·제한 |
|---|---|
| 학습 P/C/V+CPU_T, 실전 P/C+CPU_R | 모델의 P/C/V와 역할별 private parameter를 준비했다. 제품은 V-free P/C이며 실제 P/C/V 학습은 미실행이다. |
| 역할별 private latent와 공유 reader | 폭 384, Q/KV head 6/2, head dimension 64, private latent 16×384, reader 2 block×2회, SwiGLU 1024. board 64+metadata 2 토큰 encoder 2 block, record 내부 독립 encoder 1 block이다. |
| Canonical 사실과 GPU 표현 분리 | StateStore·LinePool·ObservationStore·SituationArena·TaskTable 및 의존 관계를 구현했다. GPU 표현을 회수해도 CPU 문제의 완료 기록은 보존한다. private latent를 공통 공개 기록으로 넣지 않는다. |
| 제한된 작업·저장소·중요 기록 | 모델 입력은 record 128·후보 256·이탈 지점 128 상한이며 required critical record 누락을 거부한다. 상황·line·관측·작업·큐·실행의 별도 유한 한도를 지킨다. |
| 실제 shared P/C reader 소유 | `public_memory`+`shared_pc_if` 두 session export와 native 소비 경로를 구현했다. 공유 reader·후보 임베딩 initializer는 outer scope 한 벌이며 private 초기 latent·4 FFN·head는 6개 ONNX `If`로 hard route한다. |
| 공개 K/V 재사용과 GPU 상주 | 역할 중립 key와 cache-on/off 수치 대조 접점, bounded host page bank 및 device K/V/I/O binding 경로를 준비했다. 현재 확인한 host page는 전체 입력 단위이며 record별 증분 인코딩이 아니다. 실제 device 상주·prepack 복제·VRAM 공유는 별도 인수 대상이다. |
| CPU/GPU 작업 겹치기와 private warm-start | 현재 native 역할 응답은 drain 후 반환하는 안전 경계다. CPU opt-in Warm의 기능·역할 namespace·취소·게임 reset·물리 수명은 아래 실제 Warm39에서 확인했다. GPU Warm·CPU–GPU overlap·속도·기력 효과는 미인수이며 기본은 Fresh다. |
| 학습·resume·모델 교체 | own-source dataset·loss·recipe·zero-step AdamW·sampler/RNG/checkpoint·V-free export를 준비했다. 실제 CPU ORT P/C 수집과 별도 training-private V→CPU_T producer의 유한 실행을 확인했다. optimizer update 및 학습된 모델의 교체 일반화·강도는 미인수다. |

현재 가중치는 seed로 만든 **무작위 초기 파라미터**다. 실제 neural forward와 결정적
`legal-order-mock`은 별도 모델 종류로 식별하며 실패 시 서로 자동 대체하지 않는다.
신경망 실행이 된다는 사실만으로 유효한 체스 지식이나 개선된 착수 품질을 주장하지 않는다.

## 모델·입력·export 식별

구체 모델 소스와 명령은 [PALS 모델 패키지](../../experiments/model-research/pals/README.md)에
둔다. Torch 2.8.0·NumPy 2.2.6·ONNX 1.19.0·ORT 1.22.0·opset 17과 Rust
`ort 2.0.0-rc.10`의 실제 소비를 별도로 검사한다. Python은 offline 준비·export·독립 참조
용도이며 제품 탐색 노드마다 Python callback을 실행하지 않는다.

신형 artifact는 `rovezero.pals-model.v2`, `layout=shared_pc_if`, `layout_revision=1`이고
모델 의미는 v1이다. 기존 v1 separate P/C·P/C/V graph는 그대로 읽고 자동 변환하지 않는다.
제품용 `--rules-profile-json`은 실제 Rust encoder가 출력한 `rz-pals-rules-fields-v1`의
필드 순서·정규화·결측 의미를 pin한다. 58개 의미 문자열의 length-prefixed SHA,
source SHA와 선언 raw·canonical SHA를 기록한다. 의미 pin은 학습 입력 적합성의 증명이
아니므로 이를 별도 선언·데이터 검사 없이 학습 호환으로 승격하지 않는다.

P/C batch는 하나의 scalar 역할만 가지며 `If` condition은 CPU BOOL이다. 입력은 정확한
board·이력·record revision·모델 epoch·후보 순서·mask를 식별한다. 승격은 공통 Move16의
`0=none, 1=queen, 2=rook, 3=bishop, 4=knight`다. output은 요청 후보 순서와 차례 관점
WDL을 유지한다. P의 0 divergence padding은 의미 출력에서 제거한다. V 전용 head·latent가
제품 graph에 없음을 검사한다. raw logits·공개 K/V는 절대 `1e-4`+상대 `1e-3`, policy/WDL은
최대 절대 차이 `1e-4`로 독립 참조와 대조한다.

`reader_initializer_bank`의 byte·SHA와 branch-local copy 부재는 ONNX 직렬화 단일 소유의
증거다. native optimizer의 prepack·workspace 복제나 실제 VRAM 절약을 증명하지 않는다.
CPU `If` selected-only profile, 실제 native 수치·수명, GPU 실행과 메모리 관측을 구분한다.

## 자체 데이터와 zero-step 학습 준비

[`training.py`](../../experiments/model-research/pals/src/rz_pals_model/training.py)는
`rz-pals-data/2` 시점 고정 입력 seal과 native tensor sidecar의 byte SHA·encoder·history·
epoch·record 순서·critical·합법 후보를 대조한다. 독립 등록한 collection receipt와
source registry·split을 사용하며 target을 입력에 섞지 않는다. CPU raw 평가를 임의 WDL로
바꾸지 않고, policy·WDL·이탈·V 작업 각각의 명시 mask와 context를 요구한다.

PC bootstrap은 공유 encoder·reader·후보 임베딩과 선택한 P 또는 C private expert를
준비한다. V 단계는 V private expert만 선택하고 나머지 공유 및 P/C parameter를 실제
`requires_grad` 상태로 동결한다. zero-step AdamW parameter group·recipe·유한 비용,
role별 sampler permutation/cursor, Python·NumPy·Torch RNG, 모델과 optimizer 설정을
checkpoint에 보존한다. pending gradient가 없고 steps=0인 경계에서만 저장하며,
파일 저장·hash까지 닫은 final usage receipt를 재개 시 함께 요구한다.

[`preparation_check.py`](../../experiments/model-research/pals/src/rz_pals_model/preparation_check.py)는
등록된 실제 collection을 모두 한 번씩 소비하고 CPU frozen forward·masked loss·유한값·
parameter SHA 전후 일치를 확인한다. 이 CLI는 optimizer를 생성하지 않으며 backward·
optimizer step·GPU를 실행하지 않는다. 원시 collection·모델·receipt·보고서는 소스 밖
관리 루트에 두고, 공개 문서에서는 논리 경로와 식별만 사용한다.

[`pals_collect/native.rs`](../../crates/rz-arena/src/pals_collect/native.rs)는 실제 CPU ORT
P/C 호출의 준비 입력을 dispatch 전에 봉인하고, 물리 완료·raw 응답·논리 전달/거절·탐색
소비를 각각 남긴다. 독립 source registry의 binary·checkpoint·export·runtime·모델 구성·
encoding·epoch와 실제 자산을 대조한다. C 이탈 입력은 일반 합법 후보 head의 training row로
자동 변환하지 않고 별도 divergence sidecar와 계보로 보존한다. 가상 반박·수선에 실제
경기의 승패를 붙이지 않으며, backend가 raw 출력을 반환하기 전에 거부한 값은 오류 원인과
비관측 상태로 남긴다. raw가 관측됐다고 만들어 채우지 않는다.

[`verifier_producer.py`](../../experiments/model-research/pals/src/rz_pals_model/verifier_producer.py)는
등록된 공개 parent 입력을 재사용하되 별도 private query를 V forward와 CPU dispatch 전에
봉인한다. 자체 Rust CPU_T의 조건·완료 범위·비용과 후속 자료를 private bank에 보존한다.
제품 V를 활성화하거나 V private 상태를 P/C 입력에 넣지 않으며, 미실행 작업의 비교 순위·
정보 이득·WDL 목표를 만들지 않는다. 유한 producer 실행과 실제 V 학습은 별도 인수다.

## 현재 실행 증거

아래는 2026-10-07 문서 갱신 시점에 총괄이 확인한 실행 기록이다. 로컬 검사, 정확한 SHA의
CI 결과, 독립 등록 binary의 실제 실행과 dirty-source 모델 검사를 구분한다. reference 07의
기준은 `2e07757`+dirty source이며, reference 09는 `2d96a8a`+dirty source의 15개 파일 pin이다.
reference 09를 후속 최종 통합 SHA의 인수로 바꾸지 않는다. 후속 변경은 영향 검사를 다시 한다.
GPU 05~07의 등록 실행 기준은 `5bd8c59`, CPU paired 03의 등록 source는 `134bbe1`이다.
이전 전체 workspace 검사·후속 좁은 재검사·정확한 CI run과 등록 binary 실행을 구분한다.
`e604125`의 UCI pending-close 수정 직후 검사 전 기록은 보존하고, 후속 `134bbe1` 검사와
명시적 native loader 후보 `fc59b77`의 제한된 CPU 검사를 아래에서 별도로 기록한다.
후속 `bca916d`·`f78ecf0`의 독립 GPU 수치 검사와 `b952008`의 제품 startup 변경·CPU
검사·compile-only 등록도 별도 source·binary로 식별한다. 제품 재검사와 paired 대국의
남은 인수를 독립 checker의 정상 종료로 대신하지 않는다.

| 실제 자료 | 확인된 결과 | 해석의 한계 |
|---|---|---|
| 이전 Workspace CPU/mock 검사 | dirty-source에서 1,011개 통과, 16개 ignored. | 이전 실행으로 보존한다. ignored는 미실행이며 후속 변경의 검사 결과로 대체하지 않는다. |
| `975cce4` 로컬 CPU 검사 | workspace 1,061개 통과·16개 ignored, clippy 성공, default native CLI 4개 검사 통과. | 실제 CUDA·학습·대국 및 후속 SHA의 전체 CI 성공을 증명하지 않는다. |
| `975cce4` CI 및 `b04c886` 후속 수정 | 정확한 `975cce4`의 Linux·모델·CPU binding job 성공. Windows는 Unix 전용 fixture 2개 실패. `b04c886`에서 fixture를 수정했으며 해당 CI는 모델·binding 성공, Linux·Windows 실패다. | 해당 SHA의 실패 기록을 보존하며 `b04c886` CI 전체를 성공으로 표시하지 않는다. 후속 수정과 정확한 `5bd8c59`의 성공은 별도 행으로 기록한다. |
| `6979bdf` 전체 검사와 `5bd8c59` 후속 재검사 | `6979bdf` workspace 1,074개 통과·16개 ignored. 후속 `5bd8c59`의 optional CUDA launch payload boxing 2줄 수정에 대해 arena 69개 재검사와 전체 clippy 성공을 총괄이 확인했다. | 전체 workspace 실행 SHA와 좁은 재검사 SHA를 합치지 않는다. ignored는 미실행이며 GPU 종료·실제 학습·기력을 증명하지 않는다. |
| 정확한 `5bd8c59` CI | [CI run 37594871853](https://github.com/daejunnom/RoveZero/actions/runs/37594871853)의 Linux·Windows·모델·CPU bindings 4개 job 모두 성공을 총괄이 확인했다. | 해당 SHA의 CI 성공이다. 실제 CUDA 프로세스 종료, paired 대국과 source 밖 자료의 실행 인수를 대신하지 않는다. |
| `134bbe1` UCI 검사·CI | 후속 UCI all-feature 검사의 실행 결과 합계 249개 통과·workspace all-feature clippy 성공. [정확한 CI run 37597576640](https://github.com/daejunnom/RoveZero/actions/runs/37597576640)의 Linux·Windows·모델·CPU bindings 4개 job 모두 성공을 총괄이 확인했다. | `uci-final-08.log`와 `clippy-final-08.log`의 종료 0을 별도로 대조했다. GPU 05~07 종료 실패·paired 03 provider identity gate를 대신 인수하지 않는다. |
| `bca916d`·`f78ecf0` CI와 CPU 검사 | 총괄은 정확한 `bca916d` CI 4개 job 성공, `f78ecf0` all-feature 검사 1,092개 통과·16개 ignored·clippy/default UCI 검사 종료 0을 확인했다. [정확한 `f78ecf0` CI run 37604692688](https://github.com/daejunnom/RoveZero/actions/runs/37604692688)도 Linux·Windows·모델·CPU bindings 4개 job 모두 성공했다. | 각각의 source 검사다. GPU checker의 실행·제품 startup·paired 최종 저장은 별도 인수하며 ignored는 미실행으로 유지한다. |
| `binary-registration-06` | compiler-artifact JSON의 target·profile·features·binary SHA/크기와 `rz-uci-build-capability/1` metadata를 등록했다. source는 `5bd8c59`, UCI binary는 `7b283fea62af3518275b73bd53f6a15aadbb99d789f20f9ec04d4bcda7ab5a03`, CUDA numeric binary는 `a07495dec676c4fefd3d3bc6b367c746ee08395e6b46aced22181524968b4fd5`다. | compile capability만 확인한다. 영수증의 runtime/model loaded는 false이며, `--all-features` 명령 이력이나 source SHA만으로 다른 경로의 나중 바이너리를 같은 feature 빌드로 취급하지 않는다. |
| `model-reference-07` | wrapper 영수증 `success`, CLI exit 0. 모델 13개+학습 준비 20개, 총 33개 검사 통과. | 초기화 자산의 CPU 검사다. 실제 학습·GPU 결과가 아니다. |
| `actual-collection-01` frozen 검사 | 실제 9행 모두 한 번씩 소비. P 9/C 0. policy 4행 활성·5행 mask, WDL 9행 미관측 mask. parameter SHA 전후 동일, optimizer 미생성·backward 없음·steps 0. | 실제 C 조건부 collection, V context, 신경망 collector와 학습을 검증하지 않았다. C head 숫자 fixture와 실제 C 데이터는 별도다. |
| Native CPU PALS probe | 물리 완료 NN 입력 64개, 탐색 소비 32개를 구분한 유한 probe 통과. | 기력·처리량 향상이 아니다. 완료 입력을 모두 유효한 탐색 소비로 세지 않는다. |
| `cpu-numeric-20261007-975cce4-02` | 등록 source `975cce4`, exit 0·6개 참조 사례 통과. private repeat·cache/fresh 동등성·새 게임 cache/물리 ACK·shutdown 확인. 완료 NN 입력 20개 = public 6+role 14, 전체 3.331초, cgroup peak 284,164,096 bytes, OOM 0. 새 게임 뒤 host page reserved/pinned bytes·entries 모두 0. | CPU FP32 수치·수명 증거다. host page는 `whole_input`, allocator peak와 native resident parameter·prepack 공유는 unknown이다. GPU 성능·VRAM·강도 증거가 아니다. |
| `own-onnx-collection-01/own-onnx-01` | 독립 immutable collector binary `2444980f189b1b7d1d99ed9e71339492dded44bf268f1e7d6793df52d5756983`로 실제 CPU ORT NN 수집. 1게임·2 ply·6행, CPU 966 nodes/8 jobs, NN 완료 16개 = public 8+role 8, 역할 탐색 소비 8개. finish의 물리 shutdown·buffer 해제 확인, in-flight 0·quarantine 없음·학습 0. | 2 ply 제한의 결과는 unknown이고 value 6행 모두 mask다. C/Repair 및 divergence 계보를 실제 경기 승패나 전술 증명으로 채택하지 않는다. 독립 binary 실행을 최종 소스 SHA 검사로 합치지 않는다. |
| `model-reference-09` | 등록 collection 6행을 정확히 한 번씩 소비(P 4/C 2). policy 활성 0·WDL 활성 0·WDL mask 6, parameters 전후 동일·optimizer 미생성·backward 없음·steps 0. 별도 private V producer는 자체 CPU 신규 102 nodes, dispatch 1·실제 checks 2·후속 label 1; 부모 입력·weights 불변. | 현재 자료의 masked loss가 0이라는 결과는 학습 개선이 아니다. V의 비교 순위/task loss 목표는 0행이며 private-only, native 제품 V는 비활성이다. 기준 `2d96a8a`+dirty 15개 source pin과 최종 SHA를 구분한다. |
| GPU 시도 02 | 모델 실행 전 `native.missing_mapping`에서 실패. 실패 자료 보존. | PALS 신경망 CUDA 수치 실패로 해석하지 않는다. 입구 mapping 감사의 시점 결함은 후속 수정에서 분리했다. |
| Mapping 시점 수정 | dependency-only 입구 → 완료된 첫 native Run 이후 full audit. backend CPU seam 7개와 all-feature checker 빌드 통과. | 최종 audit 실패도 후속 실행을 막는다. CPU seam은 실제 CUDA 인수가 아니다. |
| GPU 시도 03 | 입구를 통과한 뒤 session 초기화에서 CPU EP 배정과 fallback 금지 충돌. 후속 네이티브 abort, exit `-6`. cgroup peak 3,305,578,496 bytes, OOM 0. | NN Run 이전 실패다. CPU 배정 노드와 종료 오류의 원인은 추가 확인 중이며 VRAM peak는 미관측이다. 기존 pin과 자원 한도를 조용히 바꾸지 않는다. |
| `gpu-numeric-04` | source `b04c886`, CUDA-control 실행 exit 1·77.625초, cgroup peak 3,397,308,416 bytes·OOM 0·cleanup 확인. 첫 NN Run 전 public 220노드(CPU 51/CUDA 169) 배치 gate 통과, shared P/C 453노드(CPU 61/CUDA 392) gate 거부. 등록 inventory와 CPU Gather 계열 4개 및 CUDA `MemcpyFromHost` 1개의 불일치를 보존했다. NN Run 0, 종료 후 GPU 사용 0으로 복귀를 총괄이 확인했다. | 해당 실행의 pre-Run 실패는 그대로 보존한다. 후속의 명시적 CUDA transfer 근거·등록과 05의 수치 보고서를 이 실패 기록에 소급하지 않는다. 임의 whitelist 확대·CPU NN fallback·inventory 재분류로 성공 처리하지 않는다. |
| `gpu-numeric-05` | 등록 source `5bd8c59`와 CUDA numeric binary로 실제 실행했다. `numeric.json`은 6개 CUDA FP32 사례·private repeat·cache/fresh·새 게임·물리 ACK·shutdown 통과를 기록했다. 완료 NN 입력 20개 = public 6+role 14. 그러나 보고서 저장 뒤 `malloc(): unsorted double linked list corrupted`로 프로세스 exit `-6`; execution은 failed, 전체 173.926초·cgroup peak 3,693,518,848 bytes·OOM 0·cleanup 확인·종료 뒤 GPU 사용 0을 총괄이 확인했다. | **GPU 실행 전체 인수는 실패다.** 수치 보고서의 완료/ACK와 프로세스의 정상 종료는 다른 증거다. VRAM peak·native allocator peak는 unknown이며 heap 오류 원인을 확정하지 않는다. CPU 자료나 후속 재실행으로 원래 실패를 덮지 않는다. |
| `gpu-teardown-06` GDB 진단 | 05와 같은 source·binary를 GDB의 기본 ASLR 비활성 조건에서 실행했다. stdout은 inferior의 정상 종료, stderr는 `No stack.`을 기록했다. execution의 GDB exit 0, 전체 141.735초·cgroup peak 3,840,204,800 bytes·OOM 0·cleanup 확인. | 상태는 `diagnostic_only_not_acceptance`다. 총괄은 진단 controller의 exit 1을 의도한 비인수 반환으로 확인했다. GDB·ASLR 조건이 달라 05의 수정이나 정상 실행 인수로 취급하지 않으며 heap 오류가 재현되지 않았다는 사실만 남긴다. 후속 07은 별도 행으로 기록한다. |
| `gpu-teardown-07-aslr` GDB 진단 | 05/06과 같은 `5bd8c59` checker binary·자산 pin에 ASLR을 켰다. 수치 보고서는 완료 NN 입력 20개 = public 6+role 14와 물리 ACK·shutdown을 기록했으나 inferior가 SIGABRT로 중단했다. stack에는 `libcudnn_engines_precompiled.so.9`와 `__run_exit_handlers`가 있고 stderr는 heap 손상을 기록했다. 전체 140.258초·cgroup peak 3,877,982,208 bytes·OOM 0·cleanup 확인. | GDB 자체 exit 0과 native inferior의 실패를 구분한다. 상태는 `diagnostic_only_not_acceptance`이며 **정상 GPU 종료 인수는 실패다.** 종료 중 cuDNN 호출 경로를 확인했지만 최초 메모리 손상 위치·책임자를 이 stack만으로 확정하지 않는다. |
| `fc59b77` 명시적 native loader 후보 | 총괄이 별도 명시 선택 후보를 커밋·push하고 all-target/all-feature CPU 검사 종료 0(eval unit 63개·loader 18개 및 해당 integration 검사), workspace all-feature clippy 종료 0을 확인했다. | 전체 test 수 합계는 주장하지 않는다. 이 CPU 검사 시점에는 새 후보의 GPU 실행이 인수 전이었다. 후속 08/09는 별도 행으로 연결하며 기본 선택을 유지한다. CPU 검사만으로 05/07 heap 오류의 수정·정상 native 종료를 주장하지 않는다. |
| `gpu-numeric-08-shim` 독립 checker | source `bca916d`·binary `1dee8ab061c60f78262460ccfbeb5ebb10f64fc244ad6b3d16e196898a873636`의 CUDA FP32 6사례·물리 NN 완료 20개(public 6+role 14)·runtime mapping ACK·물리 shutdown을 확인했다. execution passed·exit 0, 169.917초, cgroup peak 3,679,145,984 bytes·OOM 0·cleanup 확인. | 이 등록 조건의 독립 검사는 정상 종료했다. VRAM peak는 unknown이며 제품 UCI·arena GPU 실행 인수가 아니다. 이전 05/07 abort의 최초 원인을 확정하지 않는다. |
| `gpu-numeric-09-shim` 독립 checker | source `f78ecf0`·binary `f341e79407f2631d4b468b0cad0b208254abcc17ca5b5247dc34a8b04b76b28a`의 CUDA FP32 6사례·NN 완료 20개(public 6+role 14)·물리 ACK·shutdown 확인. execution passed·exit 0, 162.483초, cgroup peak 3,680,698,368 bytes·OOM 0·cleanup 확인. 총괄은 독립 메모리 검토 PASS도 확인했다. | 독립 GPU 수치·수명 검사의 PASS다. VRAM peak는 unknown이며 제품 startup·취소·새 게임·paired 대국을 대신 인수하지 않는다. 정상 실행을 과거 heap 손상의 최초 원인 제거 증명으로 해석하지 않는다. |
| `gpu-uci-smoke-09` 제품 startup | source `f78ecf0`·제품 binary `897c43cc17da8f79379098d623f8ac0c8139bd7687be8c35b1e7085a40a5434f`에서 `uciok`·ready 전에 stdout이 닫혔다. exit 2·97.414초, cgroup peak 3,600,363,520 bytes·OOM 0. stderr는 `PhysicalCompletionUnknown`, 물리 shutdown 실패와 startup mapping ACK `None`으로 인한 receipt publication 실패를 기록했다. | **제품 GPU startup 인수는 실패다.** transcript는 `uci` 송신 한 건이며 예정된 go 2회를 실제 실행으로 세지 않는다. 이 자료에서 정상 완료 NN 수·provider 영수증을 확정하지 않는다. 종료 뒤 GPU 0MiB 복귀는 총괄이 확인했으나 정상 NN 완료·startup과 다른 증거다. |
| 유한 runner 종료 검사 | clock+reap combined patch를 clean upstream에 적용한 별도 binary 빌드와 production-method syscall seam 14개 통과. | 실제 paired 대국은 별도 인수다. 원래 clock-only binary·patch·과거 자료는 보존한다. |
| `own-paired-cpu-01` 감사 정정 | 원래 입력·실행 자료를 수정하지 않은 append-only 정정으로, 복사한 UCI hash는 정확하지만 실제 binary에 `onnx-cpu` feature가 없었음을 확인했다. identification preflight에서 exit 1, provider·NN·readiness·게임 모두 시작 전이며 게임/NN 수는 0이다. | 대국·기력 실패로 집계하지 않는다. pair/process 영수증 생성 전 실패여서 실제 native cleanup 시간은 미관측이다. 결합 process/cleanup gate 문구만으로 물리 cleanup 실패라고 판정하지 않는다. |
| `own-paired-cpu-02` 준비 | `binary-registration-06` compiler proof와 compile-only capability를 입력 등록 전에 재확인했다. lock·opening·prepare-only 3개 명령 exit 0, private snapshot 9개를 준비했다. 준비 영수증의 engines started/NN ready는 false다. | 새 유한 CPU paired 기능 실행의 준비 자료다. 경기 실행·시계·PGN·정상 종료·강도 결과는 별도 실행 영수증이 도착할 때까지 미인수로 둔다. |
| `own-paired-cpu-02` 실제 preflight | 준비 이후 실제 readiness process는 exit 2·1.017초, `group_cleanup=gone`·errors 0으로 종료했다. stderr의 `UndeliveredDiagnostics`에는 PALS rounds·CPU nodes·소비 role 모두 0이다. supervisor의 service exit는 1, 회수 완료, 전체 16.343초 = service wait 10.808초+recovery 5.520초 등으로 기록했다. 경기 0·경기 PGN 없음. | readiness 실패이며 기력 대국 결과가 아니다. supervisor 시간은 실제 경기 시계·native cleanup 시간이 아니다. 회수된 process 영수증의 cleanup 276,164ns와 group 종료 증거를 별도로 보존한다. 준비된 opening PGN을 경기 PGN으로 세지 않는다. |
| `e604125` UCI 수정 | stop 이후 pending 착수·진단을 닫고 pipelined quit을 처리하는 후속 소스 수정을 총괄이 커밋·push했다. | 최초 기록 시점은 후속 root Cargo 검사 전이었다. 후속 `134bbe1` 검사·CI 성공은 위 별도 행으로 연결하며 이전 `5bd8c59`의 결과를 소급하지 않는다. 새 바이너리·readiness·paired 실행은 별도로 인수한다. |
| `own-paired-cpu-03` 실제 paired 실행 | source `134bbe1` 등록 binary로 120초+1초·흑백 교환 두 판을 실행했다. Rules PGN 감사에서 own CPU가 흑·백 모두 체크메이트로 승리했고 시계 감사 오류는 없다. Fastchess exit 0·group gone·cleanup 확인·process errors 0. supervisor service exit 1·회수 완료·전체 494.462초. | native provider 감사의 `native model/adapter/epoch identity differs` 때문에 integration gate가 거부됐다. 실행 영수증은 execution ready/strength eligible false·scored games 0이다. 두 판의 PGN·승패를 보존하되 정상 native integration 인수·Elo·모델 승격으로 사용하지 않는다. runner 종료만으로 NN provider identity·물리 종료를 대신 증명하지 않는다. |
| `own-paired-cpu-04` 실행과 append-only 감사 | source `bca916d`의 원래 production Debug에서 두 판의 Rules 체크메이트·시계·native provider gate 통과·scored 2를 관측했다. own CPU가 흑·백 모두 승리했다. 그러나 원래 operation은 `native evidence exceeds reserved bytes`로 최종 persistence exit 1이며 canonical pair receipt·Core가 없다. | 감사는 원본을 수정하거나 성공 영수증을 재생성해 저장하지 않았다. preflight부터 필수 최종 영수증까지의 전체 시간은 unknown이다. 64KiB 개별 pretty cap 대비 진단 재구성 67,622 bytes·초과 2,086 bytes는 Rust typed 재직렬화 미확인 값이며 전체 16MiB output cap 소진을 주장하지 않는다. 최종 저장·기력 인수는 실패 상태를 유지한다. |
| `own-paired-cpu-05` 실제 저장·Core 경계 | source `b952008`·등록 10의 CPU ORT 실행에서 흑백 두 판·Rules·시계·native integration gate를 통과했다. 첫 판은 PALS 백·own CPU 흑의 49 ply 3회 반복 무승부, 둘째 판은 own CPU 백의 37 ply 체크메이트 승리다. native receipt **68,102 bytes**가 2MiB 예약 안에서 실제 저장됐으며 PGN·Core 조립 진단도 보존했다. NN 완료 입력 12,266개·완료 role 6,133개·탐색 소비 role 6,123개, PALS session 2개의 물리 shutdown·buffer 해제·in-flight 0·quarantine 없음을 확인했다. | native 저장 수정의 실제 증거지만 **전체 실행은 exit 1**이다. Core 조립은 생성 PGN의 자유 서술 `source`가 공통 공개 HTTPS 계약과 충돌해 `core=null`·`artifact.source [InvalidSource]`로 거부됐다. 원래 실패·PGN·영수증을 변경하지 않는다. preflight부터 필수 native receipt까지 553,546ms, process cleanup 426,396ns, supervisor 전체 565.470초와 회수 3.810초는 서로 다른 관측 범위다. raw service peak 263,614,464 bytes는 Windows 전체 커밋·GPU·native allocator peak가 아니며 OOM 계수는 미관측이다. PALS failed search return 1과 취소·마감 진단도 원래 trace에 보존한다. |
| Core PGN producer 출처 연결 수정 | `project_pals_core_pgn`은 Core 참조의 `source`만 실제 locked Fastchess의 HTTPS 소스 URL로 투영한다. 원본 path·SHA·bytes·license·실행 설명과 producer 소스 commit·binary SHA를 `pgn_provenance`에 별도로 보존한다. 수정 파일 SHA는 `437e0a528bc72255e099ca1a4b9888f1b77aa0e9dca093a1a61ee3fbe2a479bd`다. 총괄은 arena all-target/all-feature CPU 검사 **221 pass·0 fail·14 ignored**, 기본 feature lib **55 pass·0 fail**, Clippy `-D warnings` exit 0과 독립 소스 검토를 확인했다. | URL은 이 로컬 PGN의 공개 다운로드 위치가 아니다. 공통 HTTPS 검증과 경기·시계·물리 수명·failure gate는 유지한다. 이 검사는 계약 fixture이며 CPU 05를 성공으로 다시 저장하지 않았다. 현재 production assembly는 live owner와 Serialize-only 증거를 요구하므로 typed 종단 replay 또는 수정된 새 paired 실행은 미실행·미인수다. 이를 JSON 재구성으로 대체하지 않는다. |
| GPU paired 06 준비의 후속 상태 | 기존 등록 10의 arena에는 위 Core 조립 결함이 있으므로 GPU 06 준비는 `superseded_prepared_not_run`·`execution_eligible=false`다. 원래 prepare/source proof는 불변으로 두고 별도 상태 파일에 새 arena 등록·새 실행 명세 필요를 기록했다. | GPU는 계속 `user-deferred`이며 새 GPU 실행은 없다. 오래된 준비 스크립트를 새 수정의 실제 실행 증거로 사용하지 않는다. |
| `f78ecf0` 증거 저장 예약과 `own-paired-gpu-05` 준비 | 고정 2MiB pair metadata 예약을 구현했다. GPU paired 05 준비 명령 2개 exit 0. 고유 input 3,143,597,770 bytes, input+output+runtime 필요량 3,227,483,850 bytes. artifact·failure recovery 각 4GiB, runtime 64MiB, output 16MiB 안에 pair metadata 2MiB를 예약했다. | execute command·engine·NN ready·GPU execution observed는 모두 false이며 GPU paired 경기는 **미실행**이다. cap·선언을 RAM/VRAM peak나 최종 저장 인수로 해석하지 않는다. CPU 04의 원래 실패를 소급해서 성공 처리하지 않는다. |
| `b952008` startup 변경·로컬 검사·CI | 실패 startup publication·진단·유한 단계 timing과 명시 CUDA startup probe 예산(1~180,000ms, 생략 기본 15,000ms)을 제품·manifest에 연결했다. 새 GPU 준비의 선택값은 120,000ms이며 실제 미실행이다. all-feature CPU 검사 1,103개 통과·16개 ignored, clippy와 default UCI·arena 검사 종료 0을 대조했다. CI의 중간 관측은 3개 성공·Windows 진행이었고, 총괄은 [정확한 CI run 37609418913](https://github.com/daejunnom/RoveZero/actions/runs/37609418913)의 최종 4개 job 성공을 확인했다(Windows 10:50:31 UTC 완료). | source 구현·CPU 검사·정확한 CI의 증거다. 새 제품 GPU startup·물리 수명·paired 저장 성공은 아직 인수 전이며 16개 ignored는 미실행이다. 새 예산 선언을 실제 실행시간이나 정상 완료로 바꾸지 않는다. |
| `binary-registration-10` | source `b952008`의 compiler-artifact 등록을 완료했다. `onnx_cpu`·`onnx_cuda`·experimental I/O binding compile feature와 CUDA path 컴파일을 기록했다. | compile-only다. runtime/model loaded와 GPU started는 false이며 모델 실행·provider ready·새 smoke의 성공 증거가 아니다. |
| `gpu-uci-smoke-10` 메모리 입구 | Windows `GetPerformanceInfo`의 커밋 여유 4,781,965,312 bytes가 등록된 최소 6GiB(6,442,450,944 bytes)보다 작아 admission이 거부됐다. systemd·GPU는 시작하지 않았다. 이후 사용자는 이번 세션 GPU 검증 보류를 선택했다. | 실제 CUDA 할당·startup 실패로 세지 않는다. 실행 중 Windows 커밋/VRAM peak는 미관측이다. 미실행·사용자 보류 상태를 성공·자동 재시도 근거로 바꾸지 않는다. |
| `gpu-uci-smoke-11` 실행 준비 | 총괄은 source `b952008`·등록 10·CAS를 고정한 바깥 launcher를 준비·freeze했다. probe 120,000ms·ready 180,000ms, 전체 240초 안에서 drain 최대 30초를 선언했다. | 준비 완료·실제 GPU 미실행·`user-deferred`다. 새 측정 조건·launcher 고정은 provider ready·NN 완료·정상 종료의 증거가 아니며 후속 사용자 재개와 메모리 입구 확인 뒤 실행 자료를 따로 인수한다. |
| 사용자 GPU 검증 보류 | 사용자가 이번 세션의 GPU 검증을 보류했다. 제품 smoke 11·최종 `b952008` source의 독립 numeric 10·새 GPU paired 06은 이번 세션에서 실제 실행하지 않는다. | 해당 항목은 `user-deferred` 후속 인수다. 준비·정적 검사와 과거 source의 08/09 PASS를 새 제품·최종 source 수치·paired 인수로 승격하지 않는다. CPU 검사·정확한 CI 4개 성공과 실제 학습 0은 유지한다. |

작은 실행·결과 JSON을 읽기 전용으로 대조한 자료의 논리 ID와 실제 파일 SHA-256은 다음과
같다. 원시 파일은 관리 루트의 `runs/pals/` 아래에 보존하며 개인 경로·command 원문·호스트
정보는 이 문서에 옮기지 않는다.

| 논리 ID (`runs/pals/` 기준) | SHA-256 |
|---|---|
| `cpu-numeric-20261007-975cce4-02/execution.json` | `561cdf38ad77ddd0236ed13cac4826002a2d0f322ba5d11035cb92da507abba4` |
| `cpu-numeric-20261007-975cce4-02/numeric.json` | `7ed25bd97c8db285684d3dbb6bbee0662f38a8e845f3cdde5c5d6fa4dfdef6dc` |
| `own-onnx-collection-01/own-onnx-01/receipt.json` | `a2a42c3786a7f9639e98c175b483c521af92fc35482e8464e6330c482b762e61` |
| `model-reference-09/validation.json` | `9257b60dd3c69a565be56473725e23bcafa5e9ff715dbfff8171d4ab28779eaf` |
| `model-reference-09/preparation-check.json` | `91984cd2d841ac6433ec93526029d0483cb0ac3c792469ab8014673fc24d4394` |
| `model-reference-09/private-verifier/receipt.json` | `8f9227c5c34b75fa6e915c99c63cce7513c483f195bcfa13cc87d445bbe7ab4f` |
| `gpu-numeric-04/execution.json` | `1453a0ca1acd9a0196f0f79d1fd1e4b6fab2f4d6bd149b5e94da27eb22f14859` |
| `gpu-numeric-04/numeric.cuda-control-primary/public-initial-placement.json` | `0d60ee96e4aa2bfd78eb3c5395e9005bde7820874c316352e79300e4a12232a7` |
| `gpu-numeric-04/numeric.cuda-control-primary/shared_pc-initial-placement.json` | `7192905c844af038dbb5cc80ac1b8f6fc1fe342b98b3986158542c69c4db9fdd` |
| `gpu-numeric-05/execution.json` | `7e2a766401fb4adb917955c0284dd1bbeeae78d9b125f53d9368c5fcbf4ffcd1` |
| `gpu-numeric-05/numeric.json` | `5921a80c2f5982cd8a90a2a8797b90c292c28f3c555110108c00ede884e3b3c2` |
| `gpu-numeric-05/stderr.log` | `2fec9196496bf208222bf35ce8032d8968a407dd34dcce54135658bc493b7463` |
| `gpu-teardown-06/execution.json` | `a5e7d7ea94739902128f49dc9e576c0e129404c91540fe27b8f67ac89e915d2b` |
| `gpu-teardown-06/stdout.log` | `b4d43b3c5775f47920f8edc1091c39d554f884a281eb959c730eee3e3668d140` |
| `gpu-teardown-06/stderr.log` | `49e1116d47bda20a27ba350b585760440aaf899c435196f894edfc207c294ea0` |
| `gpu-teardown-07-aslr/execution.json` | `6de9596b97fc4c61949a958a92cfb6fffb0994a9f5c50afbcbc2362ee83cee5c` |
| `gpu-teardown-07-aslr/numeric.json` | `25bdb7502e87dd41f3a706eb2302ef92f2885f409faa90f23452ee2fb83bceb9` |
| `gpu-teardown-07-aslr/stdout.log` | `4e461f99f68c4833b6913e6cf34e4051ed7c52a855282f6e90a7d1e546169ad0` |
| `gpu-teardown-07-aslr/stderr.log` | `8c621c067a186e39cf0ccbdac91566da29eb0e72149c1f398801907f9cbaeb76` |
| `gpu-numeric-08-shim/execution.json` | `78a06c96c02aff7df136dda0ade0f71123dca1d1c0c637302a47e3d5251189ac` |
| `gpu-numeric-08-shim/numeric.json` | `53fa4c164d9521f0c82c5290c3aceb53818ac227852e85179c466bc3c90fe071` |
| `gpu-numeric-09-shim/execution.json` | `ab03cf96f2ce3fadeb5e8729adf11c4e9a937af0e82361bca058c0bed75d6704` |
| `gpu-numeric-09-shim/numeric.json` | `48bf0e8dbb68c89d090de8b287eadbcf4aeb0abc7a79407a228365e4208c5fa7` |
| `gpu-uci-smoke-09/execution.json` | `51eebc8551946eea3ea3bc45dec5a1c452bb177b07ab04a8fa1cf623d5c60fc8` |
| `gpu-uci-smoke-09/stderr.log` | `ed9336fa078865ac1beee592e9bd9c05a655b76217331d0ee5cbff975ac7ff97` |
| `gpu-uci-smoke-09/transcript.json` | `cb0355b3571d41feb7f394c69b7a01ce389f29037660fb17e99723fc3e585639` |
| `binary-registration-06/registration.json` | `fad9dc73fddb2ea089d463a612869a4e6ca73d43eac9bf71502fa38335e594c2` |
| `binary-registration-06/build-capabilities.json` | `4cef04dfa2e6560226c2453fc0cdfb69afa49661608f0cb3bbb5a43c571e96ad` |
| `binary-registration-10/registration.json` | `b17293546dd33c419d5ccdd4861e7fec6e3a9e2d94bcef5a8c1f6772dd11012d` |
| `binary-registration-10/build-capabilities.json` | `4cef04dfa2e6560226c2453fc0cdfb69afa49661608f0cb3bbb5a43c571e96ad` |
| `own-paired-cpu-01/audit-correction-01.json` | `99cc4d96bd73ecd154228526c7035843294d51101a1c098b0265ba81c50a19c0` |
| `own-paired-cpu-02/preparation-receipt.json` | `19e331e67bb6cc4b79d815c65c9fc5ffac46f64095c0225ae60a8b262696c7fc` |
| `own-paired-cpu-02/execution-result.json` | `4cd480a94e6300a6bf139f4c1b959318075ddf8f8bf02c1c606d2790a2932d94` |
| `own-paired-cpu-02/recovered/attempt-01/external-baseline-readiness-process.v2.json` | `ecacdae96c69c2f96c12fe564830bb035971bf01e3c8822f7772065ded8d55f4` |
| `own-paired-cpu-02/recovered/attempt-01/external-baseline-readiness.stderr.log` | `23d48e310be136adae0fc8d067c6a2b8b351659c32317e931d158ac97cb9b9ce` |
| `own-paired-cpu-03/execution-result.json` | `ffa9dafedc3641f73d94b7b861776433cf7d50d7894e2d420d2118f015d3f363` |
| `own-paired-cpu-03/recovered/attempt-01/pals-arena-pair-receipt.v3.json` | `8f7c8002cd1e5c5aff8d5d84f97f9cb4ab254c663e5f7ef4e29997334955a2da` |
| `own-paired-cpu-03/recovered/attempt-01/match.pgn` | `96cabf2d1642d7e2be325b7816f923a41ecbc9b673b296ae0793f1d9ab65e0d0` |
| `own-paired-cpu-04/audit-persistence-failure-01.json` | `8251d84cc6c54e6733309a8cd5ba3ff9efccfb438093e666820c1caf96baa5e6` |
| `own-paired-cpu-05/execution-result.json` | `5a882e43b5db9fa382e5c18333ba1448f294fdfc3329822c35a86d1f55079aa6` |
| `own-paired-cpu-05/recovered/attempt-01/pals-arena-pair-receipt.v3.json` | `2cf19f0bf36f67e41e87715dcc6ee4a8c7459a7a91ca6e75ff5248749a58558e` |
| `own-paired-cpu-05/recovered/attempt-01/pals-core-assembly.v3.json` | `e753eb40400485f429f2cf3da9546ba11a70895e4e1eca85f1c3bfd6bc7fe226` |
| `own-paired-cpu-05/recovered/attempt-01/match.pgn` | `0afaad928f9b7f81bbe209cd5f827a25f0c722c2a5c20468f768730b80888f2f` |
| `own-paired-cpu-05/observed-audit-01.json` | `83f647de93e604670c1dabfc3608602c21628d610433e7f4fb15d8b5d08b032f` |
| `core-provenance-check-01/checks.json` | `2e150c34c862e559b868d58e55d27e6a82ef4427bc52d073d88d522031918531` |
| `own-paired-gpu-06/superseded-01.json` | `6fc33f3da9acc77221ed6e167439d0d32f02cc9a900772f1f89ef34ea2a248b7` |
| `own-paired-gpu-05/preparation-receipt.json` | `91528fd5b8be3c6bb1a693c34894ba2cf82c9ca46400cdd58270da2c31b9dced` |
| `gpu-uci-smoke-10/windows-commit-preflight.json` | `ba2ae8c58df480787aa261f09dd01dd5c99b7967053df83ae5389362776ebe35` |
| `workspace-check-03/uci-final-08.log` | `80ffbd12a9a30fb539719aafe9432766e0ae7a851f9398c5702e090b25c781a4` |
| `workspace-check-03/clippy-final-08.log` | `da3ea705553fbf108819b87159fbbebcbfaa02c2a277f7759fe59d823761905b` |
| `workspace-check-03/tests-startup-11.log` | `f18f4c1c3124984b92cf945baea7ef74a924a2f636b0bd084e426d55f32e390d` |
| `workspace-check-03/clippy-startup-11.log` | `59de3c77f10ebcd6e216f7137279adcd0faf20406f27967dd45bb2019176a360` |
| `workspace-check-03/default-startup-11.log` | `30c85c88ef3cb6a15ba22311e2050f490f6ab85af1bed63e610fdaaa50aa6248` |

정확한 `b952008`의 최종 CI 조회는 관리 루트의 `reports/pals/ci-b952008-final.json`에
보존했다. 직접 대조한 head SHA는 `b95200864825b2e88ca46ef86dd23ba6ff7a20a5`, 상태는
completed/success이며 4개 job이 모두 성공했다. 이 작은 조회 영수증의 SHA-256은
`97e11a1ca99e259e31229bd86181078a8b064e8868d8037e3614b15c12b17599`다.

문서만 갱신한 `0b47000d2192a7862f6274e3f0c834fd1acec728`의
[CI run 37613007601](https://github.com/daejunnom/RoveZero/actions/runs/37613007601)도
4개 job 모두 completed/success를 확인했다. 조회 영수증은
`reports/pals/ci-0b47000-final.json`에 보존하며 SHA-256은
`dc236278403e7001399e987c6bc30d3500936b57d76e3e3899533ced4b4138c9`다.
이 CI와 CPU 05의 실제 실행 source `b952008`·등록 10은 구분한다.

## 남은 인수와 진행 순서

| 순서 | 필요한 확인 | 현재 상태·책임 |
|---|---|---|
| 1 | 독립 checker의 GPU 수치·mapping·물리 shutdown과 과거 heap 실패 경계 | 08/09 독립 checker의 정상 exit 0·수치/물리 ACK PASS를 확인했다. 이전 02~07 실패·GDB 조건은 보존하며 05/07의 최초 손상 위치는 확정하지 않는다. 현재 제품 기본 선택과 명시 profile을 구분하고 단독 결과를 제품 startup 인수로 확대하지 않는다. |
| 2 | 제품 UCI의 startup·GPU 수치·취소·늦은 완료·drain·buffer 수명 | 제품 smoke 09는 `PhysicalCompletionUnknown`과 실패 영수증 publication에서 실패했다. `b952008`에서 진단·유한 timing·명시 startup 예산 지원을 구현하고 검사했다. 현재 GPU 준비값은 120초이며 생략 기본값은 15초다. smoke 10은 Windows 커밋 입구에서 GPU 시작 전 거부됐다. 새 smoke 11은 준비 완료·미실행·`user-deferred`이며 후속 실제 startup·물리 수명 자료가 필요하다. |
| 3 | 최종 source 독립 GPU 수치·제품 자원·실패 증거의 저장 | 등록 10은 compile-only다. 이전 08/09의 cgroup peak·OOM 0을 기록했으나 VRAM peak는 unknown이다. 최종 `b952008`의 numeric 10과 새 제품 자원·startup/ready/종료·실패 영수증·저장은 `user-deferred`다. 이번 세션에는 새 GPU 실행을 하지 않고 후속 인수에서 등록 조건·메모리 입구를 다시 확인한다. |
| 4 | 실제 NN 기반 own collection 및 C 이탈·V task context, frozen epoch·mask·split·누출 | CPU ORT collector 6행과 frozen preparation 전체 소비, 별도 private V→CPU_T 유한 producer 실행을 확인했다. GPU collector, C divergence head의 명시 학습 context, 유효한 정책·결과·작업 효용 목표와 더 넓은 split/holdout 자료는 별도 인수다. |
| 5 | 후속 record별 증분 인코딩·device warm-start·CPU/GPU overlap과 실제 공유 메모리 실험 | 확인한 host bank는 whole-input cache이며 현재 fresh 역할 계산을 기준으로 한다. record별 증분 인코딩·warm latent·overlap과 native prepack/VRAM 효과는 후속 별도 실험·미인수다. 의미 보존 E와 근사 A·스케줄 S를 구분하고 새 고정 작업량·비교 질문·자원·peak 관측을 등록한다. |
| 6 | V-free PALS+Own CPU_R의 유한 paired 실행·시계·PGN·native 최종 receipt/Core | CPU 01/02 입구 실패와 03 identity gate 실패를 보존했다. CPU 04는 경기·native gate를 통과했지만 최종 persistence 실패·receipt/Core 부재다. CPU 05는 2MiB 예약에서 native receipt 68,102 bytes 저장을 실제 확인했으나 Core PGN 출처 계약 오류로 전체 exit 1이다. 원래 자료를 보존하며 생성 증거와 공개 producer 출처의 접점을 따로 검증한다. GPU paired 05 metadata 준비 뒤 경기는 미실행이고 새 GPU paired 06은 `user-deferred`다. 원래 실패·unknown 시간을 덮지 않고 두 판으로 Elo를 확정하지 않는다. |
| 7 | 정확한 통합 SHA·CPU CI·독립 GPU·제품·paired 인수 연결 | 이전 source 검사와 `f78ecf0`/`b952008` 정확한 CI 4개 성공, `b952008` CPU 1,103개·clippy/default 종료 0을 연결했다. 08/09 독립 GPU PASS와 제품 09 실패·10 GPU 미시작, CPU 04 저장 실패를 보존한다. 새 제품 11·최종 source numeric 10·GPU paired 06은 `user-deferred`이며 정적 준비를 실제 인수로 승격하지 않는다. |

능동 CPU 대체 응수의 후보·반박·수선 연결, evaluator identity를 포함한 근거 namespace,
실제 착수 뒤의 완료 근거·paused 작업 재개, CPU adapter 교체 경계도 별도로 인수한다.
노드가 남아 있다는 사실을 새 root에서 작업을 유효하게 재개했다는 증거로 바꾸지 않는다.
실제 NN collector와 학습 전용 V→CPU_T 유한 dispatch producer는 구현했고 위 제한된 CPU
실행 자료를 확보했다. 기존 CPU/mock collection이나 loss fixture를 그 실행 자료로 바꾸지
않는다. NN 물리 완료·role 소비·CPU 신규 작업·실전 경기 결과·학습 목표의 관측 범위가 다르며,
모든 목표가 mask인 frozen forward 성공을 학습 준비 데이터의 유용성이나 기력으로 표시하지 않는다.

GPU 05의 heap 오류는 보고서의 수치·물리 ACK만으로 원인을 확정할 수 없다. 등록된 source·
binary·자산·runtime·placement 조건과 실패 자료를 보존하고, teardown의 실제 경계와 오류
발생 시점을 좁혀 확인한다. GDB·ASLR 변경 후의 미재현도 원인 제거를 증명하지 않으며,
07의 cuDNN exit-handler stack도 최초 손상 위치를 확정하지 않는다. 명시적 loader 조건의
독립 checker 08/09는 정상 종료했지만 그 사실만으로 원래 heap 손상 원인 제거·제품 startup·
모든 실행 경로의 정상 수명을 확정하지 않는다.
조건 변경·CPU fallback·무한 재시도로 원래 실행을 성공 처리하지 않는다.

실제 optimizer 학습은 이 목표의 남은 필수 실행에 포함하지 않는다. 이후 학습을 진행할
경우 own-source 데이터·권리·예산·role 순서·frozen epoch·holdout과 checkpoint 조건을
다시 확정한다. 공개 모델 자산이 아직 미학습이라는 사실은 문서와 실행 영수증에 유지한다.

PALS는 기존 PUCT·LC0와 모델·탐색을 바꾸는 구현이다. **자료형 변경 등 의미 보존 변경에
사용한 시간·메모리 5% 문턱은 이번 PALS 도입의 진행·통합 조건이 아니다.** 후속 의미 보존
E 최적화가 필요하면 해당 변경의 baseline·peak·작업량·채택 기준을 별도로 등록한다.
현재 비교는 선언한 모델·탐색·runtime 전체 또는 통제한 단일 축의 질문에 맞춰 해석한다.

## 교체와 복구 지시

모델·Own CPU profile·외부 상대를 각각 독립적으로 선택한다. 새 모델은 typed 역할 입력·
후보 순서·WDL·head·cache 의미와 실제 자산 pin을 먼저 검사한다. 새 CPU는 지원 질문·
조건·완료 범위·score 관점과 재개 capability를 맞춘다. 외부 상대의 UCI·옵션·시계·process
종료는 arena가 소유하고 PALS 내부 입력이나 역할 head를 요구하지 않는다.

미지원 형식·계약·provider·pin 불일치, 할당·물리 완료 실패가 발생하면 명확한 오류로
중단하고 실패 자료를 남긴다. session·buffer 완료가 불명확하면 격리하고 새 요청에
재사용하지 않는다. cache 재계산은 canonical 기록과 동일 입력에서 수행하며, cache
회수가 CPU 검사 재실행이나 완료된 사실 손실을 요구하지 않도록 검증한다.

복구는 등록된 이전 바이너리·자산·profile 및 기존 V1/V2·PUCT 또는 Own CPU의 명시적
실행 선택으로 한다. PALS 오류 뒤 숨은 mock·LC0·Stockfish fallback으로 정상 성공을
만들지 않는다. 계약·정확성·수명·대국 증거와 다른 작업의 WIP를 보존하며 총괄이 인수한다.

## 2026-10-07 후속 진행 — CPU06 실패와 CPU 준비 수정

이 절은 위 기록을 보존한 후속 상태다. `30f2a0d`의 완료 감사와 등록 11의 실제 CPU06,
후속 개별 수정·CPU 모델 준비 검사를 서로 구분한다. **전체 목표는 아직 미완료이며,
CPU06은 integration 실패, GPU 검증은 사용자 보류 상태다.** 아래 좁은 검사나 원시
프로세스 종료 성공을 최종 Core·대국·GPU·학습 인수로 바꾸지 않는다.

### CPU06의 실제 결과와 두 실패 경계

`own-paired-cpu-06`은 등록 11의 source
`30f2a0dc6d49a79fd73fb27adc84209ea0b60a53`로 실행했다. Ponder off, 엔진별 선언
CPU 2·동일 affinity `0,2`, GPU 없음, 120초+1초 피셔, 흑백 교환 두 판의 기존 조건을
유지했다. `failure-review-02.json`과 supervisor 결과를 읽기 전용으로 대조했다.

| 관측 | 실제 결과 | 인수 범위 |
|---|---|---|
| runner 프로세스·정리 | native process exit 0, process group `gone`, `cleanup_verified=true`, unresolved owner 없음. | runner 종료·정리 증거다. 서비스 전체·NN·시계·Core 성공과 동일하지 않다. |
| 서비스·자료 보존 | service exit 1, `recovery_complete=true`, `integration_checks_passed=false`. native pair 영수증과 원래 PGN·로그를 보존했다. | 실패 실행과 회수 성공을 함께 기록한다. 원본 실패를 소급 성공으로 바꾸지 않는다. |
| 첫 경기 | 백 `pals-onnx-cpu`, 흑 `own-cpu`, PGN 결과 `0-1`, PALS 백의 시간패 초과 5ms. | 실제 패배 기록이다. 두 번째 판과 함께 보존하며 표본 부족이나 감사 오류를 이유로 유리하게 누락하지 않는다. |
| 두 번째 경기 | 백 `own-cpu`, 흑 `pals-onnx-cpu`, PGN 결과 `1-0`, PALS 흑의 시간패 초과 7ms. | 실제 패배 기록이다. pilot 두 판으로 Elo·모델·탐색 승격을 확정하지 않는다. |
| PGN 감사·Core | `PGN engine loss winner disagrees with A side to move`, `scored_games=0`, Core collection `not_reached`. | 정식 집계·Core 인수가 이 실행에서 이뤄지지 않았다는 뜻이다. 두 시간패가 없었다거나 무승부였다는 뜻이 아니다. |

고정 Fastchess는 반환된 후보를 PGN에 먼저 기록한 뒤 시간패를 검사하며, 시간패 후보는
실제 board에 적용하지 않는다. 현재 arena의 재생은 마지막 늦은 후보를 실제 적용수로
읽어 실패 색과 차례를 뒤집었다. 이는 PGN 감사의 별도 정확성 문제다. **native full-clock
감사의 `engine_loss` 거부는 실제 시간패에 대한 유효한 별도 판정**으로 보존한다. PGN
감사를 수정해도 5ms·7ms 시간패가 사라지거나 full-clock·Core가 통과한 것으로 표시하지
않는다. 원시 PGN을 덮어쓰지 않고 실제 적용수와 기록된 늦은 후보의 경계를 감사한다.

### 후속 수정과 CPU 모델 준비의 확인 범위

아래 검사 수는 총괄이 확인한 개별 실행 범위다. `model-reference-10/validation.json`은
CPU provider·상태·source pin을 직접 대조했으나 개별 unittest 개수는 담지 않는다.
clean `30f2a0d` CI와 dirty-source 모델 검사, 후속 커밋의 좁은 검사를 합쳐 같은 통합 SHA의
전체 성공으로 표시하지 않는다. 이번 절은 후속 HEAD의 새 CI 성공을 선언하지 않는다.

| 변경·자료 | 확인된 구현·검사 | 남은 경계 |
|---|---|---|
| `dcb312c` | Rules의 정확한 상태·이력을 사용하는 유한 UCI 수순 재생 접점을 추가했다. 총괄이 CPU 검사 52개 통과·2개 ignored를 확인했다. | ignored는 미실행이다. 외부 checker의 실제 소비·실행 인수를 대신하지 않는다. |
| `02b01a8` | RulesTerminal 체크메이트의 승자와 차례 일관성(MF-01), 잔여 예산에 따른 collection 실제 읽기 제한(MF-02)을 수정했다. 총괄이 Rust data 검사 21개 통과를 확인했다. | 정상 ActualGame 후속 결과와 RulesTerminal을 구분한다. 새 모델·전체 학습 준비·paired 인수로 확대하지 않는다. |
| `361c782` | 성공한 CPU_T capture 뒤 quota·저장 실패에도 조건·실제 반환·원래 오류를 회수하도록 MF-03 보존 경계를 수정했다. | source 수정과 해당 실패 보존 검사를 실제 학습·GPU 또는 모든 producer 경로의 종단 성공으로 바꾸지 않는다. |
| `model-reference-10` | source `30f2a0d`+dirty의 개별 source 파일 pin으로 CPU unittest 58개, 미학습 `shared_pc_if` export와 독립 CPU 수치 검사를 총괄이 확인했다. 소형 영수증은 `status=success`, provider CPU, 학습 0 step, GPU `not_run`이다. | 초기 파라미터의 CPU 준비 증거다. source pin 없이 clean `30f2a0d` CI나 후속 커밋 전체에 재사용하지 않으며 학습·기력·제품 GPU 성공으로 승격하지 않는다. |
| `4a8a446` checker 교체 경계 | 자체·외부 report를 분리한 인터페이스와 유한 Linux UCI owner를 추가했다. 첫 search 전체 171개와 후속 checker 집중 16개가 통과했다. 후속 정적 검사의 Copy 정리도 통과했다. | 집중 검사는 이전 전체 검사의 일부와 겹치므로 개수를 더해 고유 검사 수로 표시하지 않는다. 외부 CPU_R의 PALS 탐색·제품 CLI·receipt 소비는 아직 미완료다. |
| `e330c7c` PALS 작업 마감 | 기존 soft/admission 중 이른 시각을 작업 deadline으로 전달했다. 해당 소비자 회귀를 포함한 UCI lib 133개가 통과했다. 전역 시계·물리 완료 fence는 유지했다. | 실제 두 판에서 시간패가 해소됐는지는 새 등록·실행 전까지 미검증이다. |
| `5f9f037` timeout PGN 감사 | 실제 적용수와 기록된 늦은 후보를 구분했다. PGN 29개와 native pilot 5개가 통과했다. 영향 패키지의 Clippy `-D warnings`와 workspace fmt 검사도 통과했다. | 기존 CPU06 실패를 성공·Core로 바꾸지 않는다. 새 필드는 failure-only이며 원시 PGN은 보존한다. |

### 전체 목표에 남은 구현과 별도 실험

`completion-audit-30f2a0d-02.md`의 감사 시점은 `30f2a0d`다. 감사 뒤 수정과 실행은
위처럼 추가 기록하며, 감사의 구현 공백을 과거 검사·모델 fixture로 소급 완료 처리하지
않는다. 실제 optimizer 학습 제외는 데이터·계약·학습 준비 API의 미구현을 제외한다는
뜻이 아니다.

| 항목 | 남은 구현·인수 |
|---|---|
| 외부 CPU_R checker 소비 | 대국 상대 UCI 연결·`CpuSearcher` trait·mock을 넘어, 실제 문제 조건·capability·authority·raw CP/WDL/mate/bound·deadline·generation·`stop`/`bestmove`·프로세스 종료와 실패를 내부 checker가 소비하는 경로를 인수한다. 현재 WIP를 완료로 표시하지 않는다. |
| record별 공개 K/V | 현재 whole-input cache와 독립 record encoder를 구분한다. record/chunk delta→새 page→기존 page 재사용→native join·pin·evict의 실제 소비 경로와, 표현 회수 후 완료 CPU 근거 보존을 구현·검사한다. page key 타입만으로 구현 완료를 주장하지 않는다. |
| 역할별 private warm-start | 현재 fresh 역할 latent를 기준으로 남긴다. 다음 query의 역할 상태 재사용 경로와 fresh/warm 의미·오차·실제 비용 대조는 미인수다. warm-start·device 상주·압축·overlap의 효과는 후속 A/S/E 비교이며 구조적 준비와 연구 성과를 구분한다. |
| P 동일 시작 상태 비교(DG-01) | 같은 시작 상태의 후보를 묶는 pair/context, 별도 comparison target·collation·loss API가 남는다. 일반 policy loss를 이 비교의 대체 구현으로 표시하지 않는다. |
| C native divergence(DG-02) | `pals_collect/native.rs`의 `NativePreparedContext::Divergence`가 실제 divergence 입력과 sidecar에 capture하는 경로는 존재한다. 현재 `training_admission=deferred_divergence_head`, `counterfactual_wdl=masked`이며, 남은 것은 전용 감독 context/envelope·loader 입장·target/ranking 연결이다. 별도 capture와 조건부 BCE를 후보 ranking 완료로 해석하지 않는다. |
| V 비교 감독·문제 입력(DG-03/04) | 실제 조건부 관측의 comparative utility target producer·유효한 owned ranking 자료, 같은 parent·예산에서도 다른 branch/question을 구별하는 의미 입력이 남는다. provenance SHA는 branch 표현을 대신하지 않으며 미관측 rank를 만들어 채우지 않는다. 제품 V-free 조건은 유지한다. |
| 후속 label chain(DG-05) | 선행 label 존재·동일 immutable input 귀속·causal revision·cycle/missing predecessor 거부·현재 학습 view 선택 감사가 남는다. canonical observation revision과 dataset label revision은 다른 소비 경계다. |
| 일반 dataset frozen identity(DG-06) | 등록된 game·producer별 OwnPals source·model·frozen epoch의 일관성을 일반 admission에서도 검사하는 계약이 필요하다. 단일 native collector의 보장을 임의 등록 dataset의 보장으로 확대하지 않는다. 서로 다른 두 엔진의 모델이 다를 수 있으므로, 선언된 engine roster를 구분하지 않고 game 전체에 하나의 모델을 강제하지 않는다. |
| 양의 owned target 자료(DG-07) | positive fixture loss와 실제 owned 양의 target 자료를 구분한다. 기존 6행의 all-masked 전체 소비와 V의 unknown mask는 올바른 미관측 처리이며 학습 개선이나 API 미구현의 대체 증거가 아니다. |
| 수정된 최종 종단 인수 | 새 검사·정확한 source/binary 등록 뒤 유한 typed/Core 저장·시계·PGN·실패 보존을 따로 인수한다. CPU06의 runner 정리 성공이나 과거 Core projection fixture를 새 live-owner 종단 성공으로 바꾸지 않는다. |

후속 담당의 source 확인에 따라 C divergence의 미완료 범위를 위 연결부로 명시한다.
기존 감사의 미완료 판정을 **native capture 전체가 없다는 뜻으로 확대하지 않는다.**
기존 감사 원문은 보존하며, 실제 capture·학습 입장 보류·counterfactual WDL mask와 남은
감독·loader·ranking 준비를 이 정정 기록에서 구분한다.

제품 GPU startup·독립 수치·취소·물리 drain·paired는 계속 `user-deferred`다. 이 문서
갱신은 새 GPU 실행·자동 재시도 권한을 추가하지 않는다. 원문 후속 모델 구조·압축·
warm-start·스케줄 후보도 보존하며, PALS 전체 도입에는 의미 보존 변경의 5% 문턱을
적용하지 않는다. 실제 학습과 학습된 기력 검증은 이번 목표의 제외 범위로 유지한다.

### 후속 근거 식별

작은 JSON·감사 문서의 아래 식별을 대조했다. 원시 PGN의 식별은 CPU06 failure review가
연결한 보존 자산이며 이 절에서 PGN을 재작성하거나 새 경기 결과로 변환하지 않았다.
원시 로그·개인 경로·호스트 정보는 소스 문서에 옮기지 않는다.

| 관리 루트 기준 논리 자료 | SHA-256 |
|---|---|
| `reports/pals/completion-audit-30f2a0d-02.md` | `3c827e6c48e2d83579df53a1ca2c76755518f88d0ce7fa076ebad6a19b3aea8b` |
| `runs/pals/own-paired-cpu-06/failure-review-02.json` | `9c196441d84efc080241c7de5589a3000581613089237dae2e2b6a60dc4dd799` |
| `runs/pals/own-paired-cpu-06/execution-result.json` | `1a2f9c2cfa11a16b6f4e4746836ad2038b2b4da24945f77709a50361da77dd77` |
| `runs/pals/own-paired-cpu-06/recovered/attempt-01/match.pgn` | `7b407a18753646b4517eb6ef2c972f6ca52929cc3f6c90bd23b88b531ae9674b` |
| `runs/pals/model-reference-10/validation.json` | `98b7f745e7a83facbbc2dbc384c093241608022f403e53dbc91c4df8c0ba263b` |

## 2026-10-07 후속 진행 — 현재 라벨과 외부 checker의 탐색 소비

이 절은 앞선 실패·인수 기록을 보존한 추가 상태다. GPU는 사용자 지시에 따라 이번
세션에서 보류한다. 실제 optimizer 학습은 여전히 제외하며 CPU 검사의 성공을 GPU·
학습·기력 인수로 바꾸지 않는다. PR #23은 `develop` 대상 Draft로 유지한다.

### 라벨 계보와 실제 소비자

`9c91128`은 선행 라벨의 존재, 같은 immutable input 귀속, 엄격히 증가하는 관측
sequence와 단일 causal chain을 검증한다. fork·duplicate·missing predecessor·cycle·
다중 labeled root를 거부하고 원시 행·raw digest·split은 보존한다. 최신 whole label만
현재 학습 view에 들어간다. 최신 라벨이 policy/value를 mask하면 이전 target을 자동
병합하거나 되살리지 않는다. unlabeled capture는 labeled chain의 별도 root가 아니다.

Rust/Python의 새 `rz-pals-label/1`과 current view는 확률의 f64 bit 표현을 정규화하여
같은 식별을 계산한다. 기존 input·raw dataset·split·checkpoint domain은 바꾸지 않는다.
Private V context는 공개 label seal 밖에서 별도 실제 query·control 검증을 계속한다.
현재 leaf만 읽어 원시 이력의 변경을 생략하는 입장은 허용하지 않는다.

`5ee8f1c`는 준비 프로그램과 V producer에 이 current view를 연결한다. 모든 원시
자료의 검증을 유지하면서 현재 input을 한 번만 선택하며, V의 `max_steps`는 raw row
index가 아닌 선택된 input의 순번에 적용한다. 준비 영수증은 raw count·raw hash와
current count·view hash를 따로 기록한다. 기존 읽기 경로를 실제 학습 실행으로 표시하지
않는다. 총괄의 Rust data 검사 25개와 후속 Python 전체 72개가 통과했다. Python 검사는
CPU fixture·준비·소비 경계이며 optimizer update와 GPU 실행은 0이다.

엄격한 계보 검사를 연결한 뒤 실제 collection producer의 결함도 드러났다. 기존
OwnedCpu policy와 후속 ActualGame 결과가 같은 input에 각각 predecessor 없는 labeled
root를 만들었다. 최신 `5ee8f1c` CI의 Linux/Windows 수집기 4개 실패도 이 경계를
확인했다. CPU bindings와 model CPU job의 성공은 이 실패와 구분한다. 수정 중인
producer는 실제 append에 성공한 선행 라벨의 digest만 연결하며 저장 실패 후 rows·
predecessor index를 진행시키지 않는다. 원래 실패 로그는 보존한다. 이 수정의 최종
검사·커밋·CI는 별도 후속 증거로 기록하며 이 절에서 성공으로 미리 선언하지 않는다.

### 외부 checker와 모델 WDL의 탐색 접점

`86affc1`은 search의 checker 소비와 Native 모델 값 접점을 공유한 WIP다. 외부 보고서의
cp·mate·bound·reported depth·선택적 탐색·미관측 작업량은 자체 CPU raw score나
완료 iteration, WDL 또는 Rules 증명으로 변환하지 않는다. raw 보고서와 실제 관측
작업량, 예약한 node 예산을 독립 보존한다. unknown nodes는 unknown으로 남긴다.

외부 PV를 정확한 Rules 상태·이력으로 재생한 뒤 해당 후보의 frontier를 선택한 모델의
기존 contextual WDL로 다시 평가한다. `ModelValueIdentity`와 실제 prepared input key를
검증하고 W/L 관점 반전과 draw 보존을 적용한다. 미검사 방어를 전체 게임의 승패로
확정하지 않는다. 직접 현재 상태의 Rules terminal과 제한된 경로의 terminal 전파는
`RulesTerminal`·`RestrictedRulesLine`으로 구분한다. 별도 resolver는 CP 보정·평균·
unknown의 0점 대체를 수행하지 않는다.

Native 값 조회는 기존 Proposer forward의 shared WDL을 사용한다. 새로운 candidate
value head·인코딩을 도입하지 않는다. 물리 실행 전에 만든 input key를 실제 반환에서
전달하고, 조회 완료와 탐색 소비를 구분한다. 뒤늦게 남은 시간 특징을 다시 해시하여
준비 입력을 바꾸지 않는다. 기존 단일 물리 worker·buffer 소유권·finish fence를 유지한다.

총괄의 마지막 중앙 library 검사에서 search 163개·UCI 135개가 통과했고 두 패키지의
all-target/all-feature Clippy `-D warnings`도 통과했다. 앞선 컴파일 오류와 fixture 실패를
보존했다. 검사 수는 각각 해당 실행의 수이며 과거 겹치는 검사를 합쳐 고유 검사 수나
현재 통합 전체 성공으로 표시하지 않는다.

외부 helper의 제품 CLI·등록 profile·두 owner의 독립 종료·receipt·실제 arena 인수는
후속 연결이다. `86affc1`의 search 소비 성공이 이 경계의 완료를 뜻하지 않는다. 공개
foreign WDL 결론 특징, record별 native K/V 소비, 역할별 warm-start, DG01~04·DG06~07의
남은 학습 준비 계약도 위 전체 목표 감사와 함께 유지한다.

### 후속 중앙 인수 — producer 수정과 제품 checker 수명

`0ee4669`는 실제 collection producer가 같은 input의 후속 label을 저장할 때 마지막
성공 append의 digest를 predecessor로 연결한다. 서로 다른 label owner를 독립 유지하며,
실패한 append 뒤 rows·causal index를 진행시키지 않는다. 중앙 collection 검사는 순차
21개와 네 test thread의 21개가 각각 통과했다. `5ee8f1c`의 네 CI 실패는 보존하며,
`0ee4669`의 CI run `37631162901`에서 Linux·Windows·CPU bindings·model CPU 네 job이
모두 성공한 것을 확인했다. 이 결과는 후속 dirty 소스나 다른 SHA의 CI 성공이 아니다.

`af775ad`는 외부 helper의 등록·실제 startup·게임 초기화·유한 종료를 제품 driver와
CLI에 연결한다. 기본 own 경로의 v1 identity를 보존하고 명시적 external checker에는
별도 resolver·등록 digest를 사용한다. helper는 unstarted 상태로 구성하며, 모델 준비와
driver 구성 이후 유한 시계·취소 아래 실제 시작한다. 준비 실패·영수증 저장 실패·대국
서비스 실패에서는 Native와 helper의 종료를 각각 시도하고 각 원인을 보존한다.

제품 선택은 `--pals-cpu-checker=external-uci`, `--pals-cpu-profile`,
`--pals-cpu-profile-sha256`로 명시한다. 첫 profile은 내장 NNUE Stockfish의 제한된
등록 형식이며 파일 SHA·canonical SHA·지원 옵션 검사와 실제 UCI 식별을 구분한다.
`readyok`나 설정 송신만으로 적용값·학습 이력·모델 로딩을 관측했다고 표시하지 않는다.
contextual 모델 WDL을 제공하지 않는 mock에 외부 checker를 조용히 연결하지 않는다.

V3 Native 영수증의 선택적 `cpu_checker`는 두 owner의 근거를 분리한다. 시작 전
실제 cleanup과 시작된 process의 exit·pipe drain을 구별하고, 미관측 값은 null/unknown으로
남긴다. work 관측이 실패하면 서비스 종료를 성공으로 게시하지 않는다. 외부 cp·mate·
reported bound가 Native 물리 완료나 Rules 증명을 대신하지 않는다.

소스 pin을 고정한 `rz-search`·`rz-uci` all-target/all-feature 중앙 검사 468개와 Clippy
`-D warnings`가 통과했다. 앞선 API 접점의 컴파일 실패 로그를 보존했다. 관리 build
slot은 이 검사 종료 후 정책 상한을 넘은 재생성 산출물만 회수했으며 소스·인수 자료는
보존했다. 이 회수는 테스트 성공이나 메모리 성능 개선의 대체 근거가 아니다.

`969099e`는 arena의 기존 own V3 소비자가 새 work completeness boolean을 잘못
u64로 읽는 경계를 수정한다. 해당 필드만 PALS의 bool/null로 읽고 startup의 true·
CPU 경로 혼입·잘못된 타입은 거부한다. 기존 mandatory numerical counter 조건은
유지한다. 중앙 `pals_launch` 검사 24개가 통과했다.

위 제품 단위의 CPU 성공과 실제 외부 helper·arena 인수는 구분한다. 기존 own V3의
raw resolver·자원·Core counter 조건은 외부 profile에 그대로 사용할 수 없다. 명시적
외부-helper launch 등록, helper 자산 snapshot·합산 자원·종료 영수증, 실제 Stockfish
실행과 수정 이후 paired 종단 인수는 여전히 후속 작업이다. DG06 metadata 구현도
live producer·loader·resume 입장과 구분하여 별도 중앙 검사로 인수한다.

### DG06 metadata와 실제 helper 식별의 추가 단위

`44f65e4`는 Linux helper의 spawn 뒤 실제 `/proc` PID·process group·start tick 대조에
성공한 경우에만 `ExternalProcessIdentity`를 보존한다. 종료 후 같은 역사 식별을 유지하며
own checker·지원하지 않는 호스트·spawn 이전 실패에는 None을 남긴다. nested checker
영수증에 이 식별을 추가했으며 exit·pipe drain·Native NN 완료와 독립적이다. 중앙 search
164개·UCI 152개 library 검사와 두 패키지 all-target/all-feature Clippy가 통과했다.
실제 arena의 inherited cgroup·helper PID join 인수는 이 타입 추가만으로 완료되지 않는다.

`17fc6c5`는 DG06의 첫 metadata 계약이다. 별도 domain의 producer roster·unique input
capture binding·envelope가 기존 raw dataset·split·current view와 독립 등록 pin을 묶는다.
같은 게임의 두 producer는 다른 모델을 가질 수 있지만 각 producer의 source·epoch·
등록 encoding은 모든 원시 이력에서 고정한다. 기존 input/raw/split/current seal과 label
계보를 다시 검사하며 label provenance나 차례에서 input owner를 추정하지 않는다.

NativeExact는 실제 선언된 encoding을 대조한다. PrivateCheckedDerivedQuery는 별도
query schema·encoder·parent pin을 보존하고 항상 `requires_derived_adapter`에 남긴다.
schema SHA의 형식 일치만으로 실제 private query·tensor 검사를 통과시키지 않는다.
Python metadata 경계도 independently pinned raw receipt와 actual records/registry bytes,
별도 fully validated current-view pin을 요구하며 기존 raw f64 seal을 새 표현으로 바꾸지 않는다.

소스 pin을 고정한 중앙 Rust `pals_data` 33개(기존 25개+신규 8개), stdlib Python metadata
12개, `rz-experiments`·`rz-arena` all-target/all-feature Clippy가 통과했다. Python의 이번
검사는 모델·Torch·optimizer를 실행하지 않는다. 결과 scope는 항상 `metadata_only`이며
실제 producer 사용·학습 입장·전체 목표 완료를 뜻하지 않는다.

다음 연결은 독립 등록을 검증한 producer handle, 실제 capture evidence, 기존 출력
budget 안의 roster/capture/envelope 영수증과 strict loader/resume이다. Native capture의
현재 의미는 seal-before-submit·prepaid drain-after-search이며 durable disk write-before-submit
증거로 보고하지 않는다. prepared journal 전체와 raw learning history의 unique input binding을
구분해 divergence·거절·실패 입력을 보존한다. 기존 legacy loader/checkpoint에는 새 pin을
자동 생성하여 strict 인수로 승격하지 않는다.

### 외부 CPU_R 선택과 host record projection page의 첫 연결

`3e2125b`는 명시적 `cpu_r` 선택을 V3 명세·lock에 보존한다. 기존 Own 선택의
직렬화 bytes와 canonical digest는 유지하며, 외부 profile·binary·모델 WDL resolver·
자원 및 유한 수명 선언을 별도 타입으로 검증한다. 중앙 manifest library 검사 21개가
통과했다. 현재 실제 arena launcher·helper cgroup·Core projection 인수는 연결 전이므로
외부 선택의 실행 검증은 거부한다. 선언을 읽었다는 사실을 실제 자원 적용으로 표시하지 않는다.

명시적 `enable_host_record_pages`는 기존 whole-input 캐시와 별도의 물리 projection
page 경로다. 실제 16개 FP32 feature bit와 public graph·checkpoint·encoding·game을
식별에 묶고, 전체 canonical input·관측·CPU task identity는 그대로 보존한다. missing
record를 하나의 subset Run으로 공급하고 원래 head-major 순서의 K/V와 mask를 join하여
기존 private P/C Run에 전달한다. 빈 record의 false-mask padding도 실제 zero-feature
projection을 사용한다. 같은 feature의 물리 page 공유가 관측·방문·CPU 검사 재사용을 뜻하지 않는다.

페이지 budget은 entry backing과 실제 소유 배열을 포함하며 transient reservation은
준비 입력·subset·join·출력·page 복사의 겹치는 수명을 포함한다. known completion에서만
pin과 scratch를 해제하고 unknown physical completion에서는 session·full input·subset·
page·pin·unfinished join을 같은 owner에 보존한다. ORT workspace·allocator overhead·
프로세스 peak·VRAM peak는 이 호스트 예약으로 관측했다고 주장하지 않는다.

동결한 4개 소스의 중앙 `rz-eval` library 76개와 `rz-runtime` library 29개, 두 패키지
all-feature library Clippy `-D warnings`가 통과했다. actual CPU whole/page 수치 동등성은
별도 검증 전이며 제품 CLI·Native receipt 선택은 아직 연결하지 않았다. subset Run에서도
기존 public graph는 board 66개 토큰을 다시 계산한다. page hit·encoded record slots·
physical B1 NN 입력·탐색 소비 수를 구분하며 속도·메모리 개선을 주장하지 않는다.

`record-pages-numeric-01`은 `c6ead34`와 별도 예제 source pin의 실제 CPU ONNX
정확성 인수다. 기존 PyTorch reference 6개의 whole 결과를 유지하며 page 경로도 같은
6개 reference에 대조했다. 추가 whole/page 파생 입력 15개는 0/1/2/128 records,
padding과 실제 zero-feature record, append·correction·reorder·critical·ID 이동·삭제·
P/C 순서를 포함한다. 최대 policy 절대 차이 `5.96046448e-8`, WDL `8.94069672e-8`,
K/V key `1.19209290e-6`, private latent `1.66893005e-6`는 기존 허용 오차를 만족했다.
mask는 정확히 일치했다. 파생 입력은 모델 tensor 수치 검사이며 Rules 인증·CPU 관측·
학습 목표의 증거가 아니다.

반복·reorder·critical·ID 변경에서는 public Run이 늘지 않았고, 단일 append·correction은
missing record 한 슬롯만 공급했다. clear는 NN 실행 없이 page와 witness를 비웠으며
새 게임 이후에는 실제 public Run을 확인했다. 완료 뒤 pin·subset·join scratch·prepared
input·transient reservation이 해제됐고 프로세스 exit 0과 관리 supervisor의 자식 정리도
확인했다. 예제 all-feature Clippy가 통과했다. 실제 학습은 0 step이며 GPU는 user-deferred다.

### Strict producer 소비와 제품 host page의 CPU 인수

`7e3f236`·`33121b8`은 strict producer loader와 no-step 준비 진입점을 연결한다.
독립 등록 파일·checked source·game/producer roster·actual capture bytes·raw history·
current label view를 함께 대조하며 실패 시 legacy loader로 재시도하지 않는다.
중앙 Python CPU 검사는 각각 95개·102개가 통과했다. `33121b8`의 CI run
`37647447914`는 Linux·Windows·CPU bindings·model CPU 네 job 모두 성공했다.
이 CI 결과는 이후 Rust 변경이나 GPU 인수의 성공이 아니다.

`03f4dcd`는 실제 helper ready 시점의 PID/PGID/start ticks·cgroup v2 membership·
namespace·CPU 허용 목록·직접 resource limit을 보존한다. 관측 불가는 명시적인 이유로
남기며, 직접 limit과 ancestor effective limit·peak·옵션 적용을 구분한다. 파일 종류·
filesystem·byte·thread 상한과 caller deadline 내 cooperative 창을 적용하지만 진행 중인
kernel syscall의 강제 취소 보장은 아니다. 실제 외부 helper arena 인수는 별도로 남는다.

`f66a635`는 caller의 단일 deadline 안에 분석과 stop grace를 함께 예약한다. known
pre-go 거절은 PALS task·예약·작업량을 만들기 전에 확인하고 마지막 Rules 합법 착수를
유지한다. partial·unknown·실제 dispatch 실패를 빈 CPU 보고서로 바꾸지 않는다.

`339bb9a`의 actual checked owner와 prior registration은 수집 시점에 frozen producer를
고정한다. prepared input·tensor sidecar·lineage의 실제 JSON bytes와 journal/capture를
예산 안에서 보존한다. `strict-producer-cpu-01`은 등록한 Rust CPU producer가 원시
라벨 3행·고유 입력 2개를 만든 실제 경로다. strict Python loader와 untrained P/C CPU
forward는 raw history를 보존하고 current label 2행을 정확히 한 번씩 소비했다.
parameter digest는 불변이고 backward·optimizer·GPU는 실행하지 않았다. 현재 두 행의
policy·WDL·divergence 목표는 모두 masked다. 이 결과는 actual CPU producer 소비의
인수이며 Native producer 전체·DG01/02/07의 양의 목표·실제 학습의 인수가 아니다.
외부 확인 스크립트의 roster wrapper·보고서 경로·CLI receipt 구분 오류는 원래 로그로
보존했고, 실제 수집물을 다시 만들지 않고 별도 확인 자료에서 올바른 경계를 대조했다.

`d11385b`는 외부 CPU_R profile bytes·등록 program/source/resolver·유한 helper CLI를
확인하는 준비 단위다. Own 직렬화를 유지하고 undeclared host page 선택을 기존
whole-input mode로 인수하지 않는다. profile 확인과 실제 executable·자원·owner closure
인수는 다르며 외부 arena execution/Core guard는 계속 닫혀 있다.

`80a358d`는 host record page를 명시적으로 선택하는 제품 CLI·Native owner·영수증을
연결한다. 두 owner의 종료는 같은 절대 deadline을 사용하고 만료된 유휴 Native에는
새 최종 control을 제출하지 않는다. 이미 제출한 control의 물리 완료는 먼저 확인하고,
완료 불명일 때 기존 owner·input·pin을 보존한다. 이전 registered integration의
search/UCI all-target/all-feature 487개, eval library 78개와 arena `pals_` 65개가
통과했다. 종료 경계 수정 후 affected Native 30개와 eval/UCI Clippy·제품 build도
통과했다. 겹치는 검사 수를 합쳐 고유 검사 수로 보고하지 않는다.

`host-pages-product-cpu-01`은 실제 CPU ONNX 제품에서 시작 위치·Rules 스테일메이트·
stop을 처리하고 exit 0으로 끝났다. NN input은 17개 완료·16개 탐색 소비, 취소는
1개다. 마지막 실제 `SnapshotStats`의 ordinal 20에서 page pin·active input/subset/join
backing·transient reservation은 0이며 physical shutdown과 native buffer release를
확인했다. 종료 전 stats snapshot에는 cached page 2개와 retained join 68,707 bytes가
남아 있으므로 이를 owner drop 이후의 zero 측정으로 표현하지 않는다. 상속된 cgroup
관측은 새 cgroup peak·enforcement·메모리 성능 증거가 아니다.

stop 뒤 공통 ticket의 늦은 응답에서 `Admission/Stale` 진단도 관측했다. 이는 active
ticket 종료 후 결과를 받아들이지 않는 비치명 경로로 추적됐으며 실패 반환이나 새 착수를
만들지 않는다. 정상 취소와 foreign/unknown ticket을 구분하는 진단 개선은 별도 후속
단위다. 이 분류와 Native 물리 수명 인수는 서로 대체하지 않는다.

`80a358d`의 CI run `37650932282`는 model CPU 성공, 나머지 세 job은 eval formatter
검사에서 실패했다. `rz-eval`의 edition 2021에 맞춰 `0b2ba83`에서 형식만 수정했으며,
그 SHA의 CI는 관측 전까지 pending으로 둔다. 실패 기록과 source/binary의 실제 CPU
인수 SHA를 보존한다. 이번 세션의 GPU 검증 보류와 실제 학습 제외 조건은 유지한다.

| 관리 루트 기준 인수 자료 | SHA-256 |
|---|---|
| `runs/pals/strict-producer-cpu-01/preparation-report-02.json` | `b3bb0d8c93b0d6d1d01a503261e681b22350a5f64e818c5e9f966299c5849047` |
| `runs/pals/strict-producer-cpu-01/collection/strict-cpu-01/receipt.json` | `bf47edb4420a1d94e31af47445dc215c3e7791b4f05e6c726122ce337105f29d` |
| `runs/pals/host-pages-product-cpu-01`의 실제 Native 종료 영수증 | `b40d770328385e674c174af42c6ee797c47f25b2982091d590d5e311c4cefdba` |

DG05의 label chain/current view와 DG06 metadata·CPU strict 소비는 위 후속 근거로
구분한다. 남은 비교 목표·divergence 전용 adapter·V 의미 입력·private warm-start·
외부 helper 종단·새 paired 인수와 GPU 실행은 이전 전체 목표 감사에서 계속 추적한다.

### 취소 분류, preflight 소유권과 양의 WDL 준비 자료

`b7774ad`는 실제 소유한 취소 ticket의 늦은 Progress/성공 Complete만 bounded
history로 식별한다. 이 결과는 새로운 착수·작업량·backup으로 받아들이지 않고
`CanceledSearchResultIgnored` 진단으로 남긴다. foreign/unknown/evicted ticket,
자연 완료 뒤 중복 응답과 실제 실패는 기존 typed 오류를 유지한다. 오래된 timer도
현재 active ticket을 다시 확인한다. 물리 완료 불명 fence를 이 분류로 우회하지 않는다.

`041f778`의 Native owner는 외부 checker 선택에서 identification/readiness용
preflight root를 game root와 별도로 생성·pin한다. root는 역할별 exclusive directory이며
inode·UID·mode·nofollow 조건을 재검사한다. 기존 runtime의 file·byte·depth 합산 상한에
preflight tree도 포함하며 상한을 늘리지 않는다. PALS argv는 실제 owner가 전달한 root를
사용한다. 예상 경로를 계산하는 순수 API는 파일 생성·소유권의 증거가 아니다.
Own 선택은 기존 root/argv를 유지한다. 이 변경만으로 external execution/Core guard를
해제하지 않는다.

두 소스 단위의 중앙 검사는 UCI all-target/all-feature 297개, arena all-feature
library 100개, default-feature launch 29개와 관련 Clippy·format·제품 build가 통과했다.
서로 겹치는 검사 개수를 고유 검사 수로 합산하지 않는다. `041f778`의 CI run
`37654377439`는 Windows·CPU bindings·model CPU 성공, Linux 실패다. Linux의 두
정상 producer fixture는 병렬 실행 중 실제 테스트 executable 해시 비용 때문에 기존
10초 창을 넘겼다. `a1c09b5`는 이 두 정확성 fixture에만 유한한 120초 창을 적용했다.
제품 기본값과 명시적인 timeout 실패 fixture는 유지한다. 관련 26개 검사는 four-thread
실행에서 통과했고 arena all-feature Clippy도 통과했다. 새 CI는 정확한 SHA의 실제
완료 결과로 별도 인수하며 이전 실패 기록을 덮지 않는다.

`host-pages-product-cpu-02`는 `a1c09b5`의 등록 binary로 실제 CPU ONNX 제품의
startpos·스테일메이트·stop·quit를 다시 검사했다. physical role input 17개 완료,
탐색 소비 16개와 취소 1개를 구분했다. 마지막 page snapshot의 active pin·buffer·
transient reservation은 0이고 physical shutdown·native buffer release가 확인됐다.
이전 `SharedContractError` 진단은 관측하지 않았다. 종료 전 cache page 2개와 retained
join 68,707 bytes는 정상 보존 snapshot이며 owner drop 후의 zero 관측이 아니다.
상속 cgroup 자료로 새 memory peak 비교나 성능 개선을 주장하지 않는다.

`strict-producer-positive-cpu-02`는 동일한 등록 소스의 실제 Own CPU producer로
백의 mate-in-one 상태를 수집했다. 자체 Rules가 실제 체크메이트·백 승리를 확정했고,
raw history 3행에서 current 2행을 strict independently registered loader가 각각 한 번
소비했다. 기존 untrained checkpoint의 frozen CPU forward는 두 행 모두 nonmasked
WDL 목표를 받아 유한한 양의 WDL loss를 계산했다. 파라미터·raw history·current view·
frozen admission은 불변이며 backward·optimizer 생성/step·GPU는 모두 미실행이다.
이 자료는 DG07의 **실제 owned 결과 기반 WDL 준비 경로**에 한정한다. P/C 비교 policy,
divergence·V utility·모든 양의 목표 또는 실제 학습의 완료 근거가 아니다. 외부 확인
스크립트의 trailing summary가 CLI receipt와 전체 report를 혼동한 실패를 보존했으며,
모델을 재실행하지 않고 실제 산출물의 pin·loss·불변성을 독립 확인했다.

`external-helper-product-cpu-01`은 등록된 Stockfish 19 binary를 PALS 내부 checker로
실제 시작·소비·종료한 CPU 제품 검사다. GPL source/license와 binary를 MIT 소스 밖에
분리했다. Stockfish의 pin된 source header는 GPL-3.0-or-later를 명시한다.
[공식 source header](https://github.com/official-stockfish/Stockfish/blob/edb0d9db6731067ec50ce619ff372b463bc4dd5d/src/engine.cpp).
실제 profile bytes·canonical/registration SHA·전체 checker identity·model-WDL resolver·
checkpoint epoch·encoding을 Native 시작/종료 영수증과 대조했다. ready 시점의 parent와
helper는 같은 cgroup/namespace와 CPU 0/2를 관측했고 직접 high 6GiB·max 12GiB·
swap 0·pids 128을 확인했다. 이 scope는 ancestor effective limit·전체 수명 thread/option
준수·NNUE 실제 loading의 관측이 아니다. 종료 후 사라진 transient unit에 대한 기본값
조회는 실행 policy 근거에서 제외했다.

helper의 별도 역사 `(pid, pgid, start ticks)`와 native parent PID를 연결했고, known
PGID owner의 exit 0·stdout/stderr drain·cleanup, Native의 물리 완료·buffer release를
각각 확인했다. 외부 dispatched/report/consumed는 각각 1, reported nodes는 20,
reserved node budget은 256이며 Own CPU task/node 합계는 0이다. model-WDL value call
3개는 native NN 합계에 중복 가산하지 않는다. foreign completed task·reused consumption
별도 합계는 여전히 unknown이다. `work_incomplete=true`는 실제 qnode/TT 등의 미관측을
포함하므로 owner 물리 미완료로 환산하지 않는다. 새 게임 후 `latest_attempt=null`을
task별 raw history의 완전성으로 해석하지 않는다. 이 probe는 arena preflight/Core·paired
인수를 대체하지 않으며 외부 arena guard는 후속 typed 소비 검사까지 유지한다.

| 관리 루트 기준 추가 자료 | SHA-256 |
|---|---|
| `runs/pals/host-pages-product-cpu-02`의 Native 종료 영수증 | `30bb3fe8b4244f4b44f2f47230dd5ca9a47c8bf9fc8456a1931de040acc7a481` |
| `runs/pals/strict-producer-positive-cpu-02/collection/strict-positive-cpu-02/receipt.json` | `508e4f52777aa7359282b2932a04f7b0b1a6d323457767738aa18843668bcea6` |
| `runs/pals/strict-producer-positive-cpu-02/preparation-report-02.json` | `0e1ef14593373fd38e814492e994a83843fb7df7250cd82a41ce3ef880c221cd` |
| `runs/pals/external-helper-product-cpu-01`의 Native 종료 영수증 | `8ae93d3fc1488893e8dcdb4cfb8ed88ac6c01b46a31a9c656dc2ae149b699a6e` |

실제 학습 제외와 이번 세션 GPU user-deferred 조건을 유지한다. 비교 목표·divergence
adapter·V 의미 입력·warm seed의 실제 native 소비·새 paired 실행은 계속 별도 인수한다.

### 2026-10-08 후속 단위 — private seed와 외부 helper 소비 계약

`4004e09`는 opt-in approximate private seed의 Rust 소유권 단위다. P/C별 1~2개 slot,
6144개 유한 FP32 비트, 모델·인코딩·checkpoint epoch·수치 frozen epoch·게임·탐색
세대와 exact Rules context를 구별한다. Fresh 입력·public K/V·공통 revision 0.1은
유지하고 approximate invocation에 accepted seed seal을 결합한다. 원래 취소 토큰과
마감을 staging·commit까지 보존하며, known physical completion 뒤에만 provisional
latent를 만들고 채택한다. unknown/drop에서는 owner와 pin을 격리 보존하고 실제 fence를
확인한 복구 뒤에도 해당 결과를 accepted seed로 승격하지 않는다. 예산·slot 교체
거절은 기존 valid seed를 제거하기 전에 확인한다.

중앙 eval all-feature library 92개와 all-target Clippy가 통과했다. 이 중 신규 seed
fixture는 14개이며 검사 수를 중복 합산하지 않는다. 호출자는 실제 Rules encoder와
동일 snapshot 및 backend fence를 연결해야 한다. caller completion enum이나 weak
context identity만으로 임의 tensor의 Rules 의미나 Native 물리 완료가 증명되지는 않는다.
현재 export에는 initial-latent 입력이 없어 **Native warm-start는 계속 Unsupported**다.
이 단위의 성공을 실제 warm graph 소비·오차·속도·기력의 인수로 확대하지 않는다.

`5003243`는 실제 loaded helper profile에 기반한 startup/termination 검증기,
preflight 소비 훅과 별도 foreign work receipt 타입을 추가한다. profile 파일·canonical·
registration, model-WDL resolver, checkpoint·encoding·process epoch, independent parent PID,
ready 시점 cgroup/namespace/affinity/direct limit 및 historical helper identity와 known
exit/drain을 대조한다. Native physical closure와 game reset은 별도 확인한다. 외부
작업의 dispatch·report·reserved budget·reported nodes·consumption을 Own counters와
구분하고, producer가 제공하지 않는 completed/reused 합계는 unknown으로 유지한다.
서로 다른 물리 작업 단위를 임의 부등식으로 동일시하지 않는다.

중앙 arena all-feature library 102개, experiments library 70개와 영향 all-target Clippy가
통과했다. Own V3 기록은 optional foreign projection을 생략하는 기존 직렬화를 유지한다.
이 단위에서는 실제 arena preflight override·game collector·Core assembly의 외부 실행
guard를 해제하지 않았다. typed API와 fixture를 실제 종단 인수로 표시하지 않는다.

`8d885c8`의 CI run `37657068556`과 `4004e09`의 최신 PR checks는 Linux·Windows·
CPU bindings·model CPU 네 job의 실제 성공을 확인했다. 앞선 formatter/positive producer
fixture 실패와 superseded run 기록은 보존한다. `5003243` 이후 변경은 정확한 해당
SHA의 CI 완료 결과로 따로 인수한다. GPU는 이번 세션 user-deferred이고 실제 학습은
범위에서 제외한다.
`6771470`는 `rz-pals-comparative-overlay/1`의 첫 Rust/Python 계약 단위다. 현재 입력의
raw/split/current/producer 봉인, 준비 input/tensor/lineage 실제 bytes와 parent receipt pin,
독립 criterion 및 checker namespace·조건을 연결한다. P 후보와 C 응답의 candidate-only
restriction은 서로 다른 task ID를 가지되 같은 상태·이력·epoch·revision·profile·horizon·
node budget·관점을 요구한다. 후보는 captured 합법 수의 순서를 유지한다. Partial,
cancelled, unknown, missing은 masked 상태만 허용한다. 기존 input/raw/current domain과
공통 revision은 유지하며 Python에 DG05 selector를 복제하지 않는다.

모든 audit는 metadata_only이며 requires_actual_checker_admission=true다. 선언된
completed/preference, evidence hash 또는 supplied current anchor만으로 실제 CPU 요청·
완료·criterion utility·양의 target·loss의 인수를 얻지 않는다. Divergence는 별도
auxiliary adapter가 필요하다. 중앙 Python stdlib 8개, experiments library 78개(신규
comparative fixture 8개 포함), all-target Clippy가 통과했다. 두 언어의 literal vector는
한글 UTF-8, sorted compact JSON, null, array order 및 정확한 u64 9007199254740993을
대조한다. 최초 Rust fixture의 세 문자열 타입 추론 실패를 보존하고 명시적 String
선언만 수정한 뒤 인수했다. 이후 실제 checker 증거와 no-step target 소비를 연결한다.

`4004e09`의 실제 CI run은 `37658544013`이며 네 job 모두 성공했다. 후속 helper 및
comparative source는 위 중앙 검사와 새 SHA의 CI를 구분한다. 관리 build slot은 마지막
검사 후 약 4.77GiB, 임시 child tree 정리 verified이며 기존 8GiB slot/12GiB catalog
상한을 유지했다. 이 크기는 모델 peak/RSS/VRAM 성능 결과가 아니다.

### 2026-10-08 후속 CPU 연결 — 수집·후보 검사·Warm 분기

`ab6e4d5`는 외부 helper의 실제 arena owner·독립 profile·preflight·게임별 PID·
종료·foreign 작업을 Core까지 연결하는 구현 단위다. Own 작업 합계와 외부 checker
합계는 분리하고, 미관측 완료·재사용·옵션 적용은 unknown으로 남긴다. experiments
78개, arena 105개, UCI 175개 및 CPU task CLI 3개가 중앙 검사에서 통과했다.
이후 작은 error 표현·fixture 순회 수정에는 해당 검사와 영향 all-target Clippy를
다시 적용했다. 실제 외부 helper paired arena/Core 실행 인수는 아직 남아 있다.

`506114e`는 `pals_cpu_task --candidate-only`의 별도 단일 후보 dispatcher다.
새 Own CPU engine·TT를 한 요청에 한 번 사용하며 Rules 상태·이력·합법 수 순서와
PV를 검증한다. raw scalar는 차례 관점의 미보정 단위다. captured/checker/선행 계획
식별은 caller 선언이며, CLI가 실제 실행 파일을 확인한 범위와 caller가 독립적으로
등록·관측한 범위를 구분한다. 기존 V task CLI와 schema는 유지한다.

`71ddf7c`는 독립 등록·관측 자료를 받는 비교 consumer다. 기존 strict frozen
loader·current selector를 재사용하고 실제 request·stdout/stderr·launch·선행 계획·
criterion·CPU source/binary pin을 대조한다. 전체 시간은
`report.elapsed_ms ≤ receipt.elapsed_ms ≤ caller launch.elapsed_ms`를 요구한다.
partial·취소·실패·missing·terminal·mate band·tie는 masked로 보존한다. 별도 ordinal
softplus 준비는 frozen CPU FP32 forward/loss만 수행하며 기존 policy/WDL·repair·
divergence 인수와 분리한다. 합성 fixture 12개와 독립 source 리뷰가 통과했지만,
실제 launcher·strict 재로딩·P/C nonmasked target·no-step loss의 종단 인수는 남아 있다.

`48e4fe1`는 별도 private Warm export domain을 제공한다. legacy Fresh graph와
파라미터는 보존하고 full 6144 FP32 accepted-latent 입력·명시적인 Warm Bool·P/C
역할 분기를 추가했다. bounded manifest와 graph는 한 번 읽은 동일한 검증 bytes로
audit·CPU ORT 실행한다. root17에서 신규 14개·기존 13개 검사 및 실제 CPU export와
6개 수치 사례가 통과했다. 최대 절대 차이는 policy `1.641e-7`, WDL `2.863e-7`이었다.
이는 CPU 텐서 분기·수치 대조의 근거이며 실제 Search accepted seed의 Native 재사용
인수는 아니다. legacy 제품 실행은 계속 Fresh이고 Native Warm 연결은 별도 구현한다.

실제 CPU 제품 자료도 실행별 source와 scope를 나눠 보존했다.

| 관리 루트 아래 자료 | 실제 확인한 범위 | 남은 경계 |
|---|---|---|
| `runs/pals/continuation-cpu-check-03/cpu-products-registration-20.json` | `48e4fe1`의 CPU UCI·후보 dispatcher·arena 실행 파일과 compiler feature·hash 등록 | 새 소스의 바이너리로 자동 재사용하지 않음 |
| 같은 루트 `actual-candidate-check-22.json` | 기존 실제 proposer capture의 후보 2개를 각 fresh child에서 검사. Rules·PV·단일 root·완료·전체 시간·self-image 대조 | 정식 comparative bank·critic target·양의 loss를 생성한 검사 아님 |
| `runs/pals/strict-native-comparative-capture-cpu-23/capture-execution.json` | 별도 등록한 이전 `a1c09b5` collector와 고정 untrained checkpoint의 실제 CPU P/C capture 12행 | 1게임·2 ply 기능 검사. `ply_limit`/`unknown` 유지, 비교 target 인수 전 |
| `reports/pals/ci-48e4fe1-final.json` | 해당 SHA의 Linux·Windows·CPU bindings·model CPU 네 CI job 성공 | 후속 SHA와 GPU 성공을 대신하지 않음 |

root13의 CPU graph 수치 성공 뒤 managed venv symlink 정리 실패는 실패 기록으로
보존했다. 소유 token·비활성 child tree를 확인한 좁은 복구 후 root17부터 `venv
--copies`를 사용했다. root17·21·22·23의 임시 tree 정리는 verified이고 managed
build의 8GiB slot/12GiB catalog 상한을 유지했다. 초기 WSL 시작 실패는 workload가
시작되기 전의 별도 진단이며 원인을 GPU·모델·물리 RAM으로 확정하지 않는다.

이번 세션 GPU 검증은 사용자 지시에 따라 보류한다. 실제 학습·optimizer·backward와
새 cloud 비용은 실행하지 않았다. CPU 기능·수치·소유권 결과를 GPU·속도·기력
개선으로 확대하지 않는다. PR #24의 별도 자원·Ponder 변경은 읽기 전용으로 확인했으며
이 branch의 Ponder off·기존 실행 identity에 합치지 않았다.

### 2026-10-08 후속 CPU 인수 — 실제 P/C 목표·Native Warm·외부 helper pair

이번 절은 앞선 candidate consumer·private Warm export·외부 helper 연결 뒤에
확인한 CPU 인수를 소스 단위와 실행별로 추가한다. 기존 실패·마스킹·부분 완료
기록을 유지하며 문서 인수가 실제 학습·전체 PALS Search·GPU·기력 인수로 바뀌지는
않는다. 아래 경로는 관리 생성물 루트에 상대적인 논리 경로다. short SHA는 해당
소스 단위를 가리키며 서로 다른 실행의 검사 수를 고유 검사 수로 합산하지 않는다.

#### Warm graph와 Native owner의 연결

`e287e908`은 CPU Warm graph 로더와 full accepted seed binding을 연결한 단위다.
private latent의 일부 summary를 seed 대신 사용하지 않고 기존 full seed 계약으로
입력을 결합한다. 이 로더의 연결과 실제 Search/Native seed 소비는 별도 인수다.
`d287988`은 Native warm worker·exact revocation·arena의 undeclared-Warm guard를
연결한다. 원래 Fresh domain, public K/V, exact state와 approximate seed의 구분을
유지하고, 선언되지 않은 Warm 선택으로 기존 제품 경로가 암묵 변경되지 않게 한다.

root29에서 eval library 105개, UCI 185개, arena 106개가 통과했다. 이는 해당
warm/revocation/arena 연결의 CPU 검사 근거이며 뒤의 실제 Warm harness 결과와
구분한다. root37에서는 eval 105개, search 183개, UCI 185개와 workspace
all-target/all-feature Clippy `-D warnings`가 통과했다. 겹치는 suite의 수를 합산하지
않으며 이 로컬 검사와 후속 commit의 실제 CI 완료는 별도 인수한다.

#### 실제 bounded P/C producer와 zero-step 목표 소비

`777fe7a`은 bounded candidate producer 단위다. root27의 합성 fixture 16개는
실제 candidate 생성·launch·checker 소비의 종단 인수와 별도다. root30은 실제 P와
C에서 각각 한 comparative pair를 생성했고 네 checker 호출을 각각 fresh Rust
child로 실행했다. 실행 상태는 `completed_with_targets`, elapsed는 4,739ms였다.
독립 strict 재로딩 뒤 frozen CPU forward에서 P loss `0.893405199`, C loss
`1.150353670`을 계산했다. 파라미터의 실행 전후 digest는 같은
`6d5d8fad…e39d87`이었다. 이 표기는 원본 전체 digest의 축약이며 독립 artifact
pin으로 대신 쓰지 않는다. 정확한 digest는 원본 준비 report에 보존한다.

이 결과는 **DG01의 P 후보와 C 응답 비교 목표**에 한정한 실제 target/no-step loss
소비 근거다. 해당 source의 candidate-only checker·strict frozen loader·독립 source와
launch pin을 유지한다. `completed_with_targets`를 backward·optimizer step·파라미터
학습 완료로 해석하지 않으며 divergence adapter·repair 목표·V 학습·모든 비교 목표의
완료를 주장하지 않는다. 실제 checker 네 child와 P/C pair 두 개, 계산된 loss 두 개는
서로 다른 계수이므로 같은 작업량으로 환산하지 않는다.

#### 외부 helper paired CPU02의 기능·실패 경계

`external-helper-pair-cpu-02`는 실제 외부 helper의 paired runner/Core 연결과 cleanup을
확인한 CPU 실행이다. `core_integration=true`, `cleanup=true`를 기록했으나 두 판을
완료된 강도 pair로 인수하지 않는다. 한 판은 16 ply limit의 incomplete, 다른 판은
15 ply의 mate로 끝났다. 제한까지의 기능 실행과 완료된 경기 결과를 구분한다.

failed go는 총 11개다. 그중 두 개는 reserved node budget 256에 대해 reported
nodes 451/893이 발생한 node overshoot이고, 후속 아홉 개는
`checker_not_available` 실패(6개+3개)다. 각 원본 failure를 보존하고 서로 다른
원인을 helper unavailable 11개로 합치거나 정상 자체 평가로 치환하지 않았다.
`strength=false`를 유지하며 post-unit observer가
없어 supervisor가 실패한 사실도 별도 기록한다. Core 연결·cleanup true가 node budget
준수·전체 helper 수명 관측·완료 pair·기력의 성공을 보장하지 않는다. recovered PGN은
해당 실행의 실제 경과 자료이며 원래 실패 영수증을 덮어쓰는 성공 자료가 아니다.

#### Native Warm34의 실제 CPU 역할 harness

Warm34는 public checked role harness에서 P/C의 Fresh/Warm을 독립 수행한 실제 CPU
검사다. NN count는 8, 탐색 소비는 6, accepted seed는 5였다. ValueFresh는 seed bank를
변경하지 않았고 실제 newgame fence, cancel 및 늦은 결과 거절을 확인했다. Native shutdown·
buffer release·process group 소멸과 exit 0도 각각 관측했다. 이 역할 harness의 실제
Native seed 소비를 앞선 metadata/export/loader fixture의 성공과 구분한다.

Warm34 elapsed는 4,749ms, cgroup peak는 239,513,600 bytes, OOM은 0이었다.
이 값은 해당 bounded CPU harness의 실행·관측 값이며 full PALS Search의 완성,
시간 개선·기력·GPU·다른 workload의 메모리 성능으로 일반화하지 않는다. NN count,
소비된 평가와 accepted seed는 별도 카운터이며 새 방문 또는 새 네트워크 실행의
동일한 수량으로 합치지 않는다.

Warm33의 runtime copy는 RLIMIT_FSIZE에 따른 `SIGXFSZ`로 실패했다. 이 원본
실패를 Warm34 성공으로 덮지 않는다. Warm34의 managed cleanup도 read-only parent
때문에 실패했고 원본을 보존했다. root35는 종료가 확인된 정확한 owner의 managed scratch
runtime 21,050,608 bytes만 복구·정리하고 lease 해제를 verified했다. 이는 소유 범위의
후속 복구 성공이며 원래 managed cleanup이 성공했다는 뜻이 아니다. 다른 runtime·
실행 증거·모델·보고서를 이 복구 범위에 포함하지 않는다.

후속 cleanup은 `PermissionError`일 때만 삭제 대상 내부 부모 디렉터리에 owner
write/search 권한을 보완한다. POSIX에서는 no-follow FD와 inode 검사를 사용하고,
일반 파일의 권한은 바꾸지 않는다. tree-gone·owner token·경로·링크·secret·예산
검사는 유지한다. root38의 저장소·프로세스 회귀 검사 18개가 통과했고, Warm39는
같은 등록 CPU 바이너리와 입력에서 역할 검사·known native 종료·process group
소멸·관리 scratch 정리까지 모두 exit 0이었다. 출력 47,381 bytes, NN 8·소비 6·seed
승인 5, cgroup peak 236,920,832 bytes·OOM 0을 기록했다. 이 peak는 fresh service
cgroup의 보조 관측이며 Windows 전체 커밋·VRAM·native allocator peak가 아니다.
cleanup 재검증을 위한 재실행이며 Warm34 대비 속도·메모리 개선률을 계산하지 않는다.

#### 실행별 논리 근거와 남은 인수

| 소스·실행 단위 | 관리 루트 아래 논리 자료 | 확인 범위 |
|---|---|---|
| `d287988` / root29 | `runs/pals/continuation-cpu-check-03/warm-revocation-arena-tests-29.log` | eval 105·UCI 185·arena 106의 CPU 검사 |
| `777fe7a` / root27 | `runs/pals/continuation-cpu-check-03/comparative-consumer-tests-27.log` | 합성 fixture 16개. 실제 producer 자료와 구분 |
| root30 실제 P/C | `runs/pals/actual-candidate-comparative-cpu-30/frozen-pc-ordinal-preparation.json` | 실제 P/C target, strict 재로딩, frozen loss·파라미터 불변 |
| 외부 helper CPU02 | `runs/pals/external-helper-pair-cpu-02/root-functional-audit.json` | Core 연결·cleanup과 node 초과·unavailable·supervisor 실패 |
| 외부 helper CPU02 PGN | `runs/pals/external-helper-pair-cpu-02/recovered/attempt-01/match.pgn` | 16-ply incomplete·15-ply mate의 실제 경기 경과 |
| Warm33 실패 | `runs/pals/continuation-cpu-check-03/native-warm-execution-33.json` | runtime copy SIGXFSZ 원본 실패 |
| Warm34 감독·실행 | `runs/pals/continuation-cpu-check-03/native-warm-supervisor-34.log`, `native-warm-execution-34.json` | 유한 실행·cgroup peak·OOM·종료 및 cleanup 실패 |
| Warm34 역할 결과 | `runs/pals/continuation-cpu-check-03/native-warm-actual-34.stdout.json` | public checked P/C Fresh/Warm, 실제 seed·fence·취소·물리 종료 |
| root35 복구 | `runs/pals/continuation-cpu-check-03/native-warm-cleanup-recovery-35.json` | exact stopped owner runtime만 정리·lease 해제 verified |
| root38 정리 검사 | `runs/pals/continuation-cpu-check-03/storage-correctness-38.log` | 저장소·프로세스 검사 18개와 관리 scratch 정리 |
| Warm39 최종 실행 | `runs/pals/continuation-cpu-check-03/native-warm-execution-39.json`, `native-warm-supervisor-39.log` | 역할 검사와 실제 종료, stdout/stderr 제한·정리까지 exit 0 |

`777fe7a`의 CI에서는 CPU bindings·model CPU가 성공했으며 Linux·Windows는
Clippy `nonminimal_bool`에서 실패했다. 그 실패와 해당 SHA를 보존한다. 후속
수정 뒤 `675f3a5`는 nonminimal-bool CI 수정, `0c9aaf4`는 CPU Warm harness와
known-fence reset, `f7b0070`은 실제 실패 external namespace의 admission을 새 NN
실행 전에 차단하는 단위를 각각 commit·push했다. 실제 원본 attempt와 실패 이유는
보존한다. root37의 로컬 Clippy 성공을 이 새 head의 CI 성공으로 보고하지 않으며
해당 head의 실제 완료 CI는 아직 미확인이다. 원래 Warm34의 기능 성공·managed
cleanup 실패·root35 복구를 유지한 채 root38 fixture와 Warm39 실제 bounded 재실행을
위처럼 별도 인수했다. 실제 실행 바이너리는 build32의 dirty-source 등록이며,
문서 통합 SHA나 후속 clean HEAD에서 새로 빌드한 바이너리로 표시하지 않는다.

GPU는 사용자 지시에 따라 보류 중이고 실제 training·backward·optimizer·새 cloud
실행은 하지 않았다. 이후에는 정확한 새 SHA의 CI, node 초과·helper unavailable·
supervisor의 원인별 인수, 충분히 완료된 paired
실행을 각각 진행 단위로 추적한다. P/C target 준비·실제 Warm 역할 소비가 전체
PALS Search·divergence/repair/V 학습·속도·기력의 완료를 뜻하지 않는다.

### 정확한 분기 의미 입력과 고정 검사 의무 효용의 CPU 연결

`4d98c7f`와 `ce93e4e`는 기존 private CPU task와 별도의
`--prepare-semantic` CLI를 연결한다. Rules의 실제 상태·이력·합법 수 순서,
prefix·restriction·claim-end를 준비하며 search·모델·target을 실행하지 않는다.
빈 claim은 unknown으로 유지하고 합법한 수순을 메이트 또는 전술적 참으로
승격하지 않는다. 원래 stdin·실행 이미지 hash·Rules 재생·출력의 deadline과
출력 한도를 유지한다. module 인자 비교 scope와 실제 CLI의 Linux loaded
executable inode scope를 구분한다. root41의 Rules 검사 11개와 root43의
CLI 검사 7개, UCI all-target/all-feature Clippy 및 format을 통과했다.

`87b7b4e`의 `CheckedSemanticInput`은 기존 strict frozen current parent를
소비한다. 원시 request·receipt·source·registration·결과 전 선언·common query·
독립 launch 관측을 각각 pin하며, Rules를 Python에서 재구현하거나 hash를
신경망 의미 feature로 사용하지 않는다. 기존 public encoder·query16·동결된 V
parameter를 유지하면서 실제 board·FEN 필드·known history·합법 수·prefix·
restriction·claim 순서를 1,394개의 private 토큰으로 준비한다. 단일 owner의
추가 예약 상한 11MiB는 기존 모델·checkpoint·전체 RSS 한도와 구별한다.
CPU FP32 정밀도·finite 값·미변경 parameter·gradient 부재와 활성 autocast의
조기 거절을 검사한다. 이 frontend는 새 private 입력 의미를 사용하며 기존 V
export의 입력 호환이나 학습된 품질을 주장하지 않는다.

`34dc701`의 효용 계약은 사전 지정된 동일 Rules 상태·이력·합법 수 순서·profile·
H/N/wall/output 조건의 완료 iteration 의무만 비교한다. 실제 동일 호출의 owned
resume와 요구 H 전체 완료에는 coverage 1, 정상 no-check Defer에는 0을 준다.
partial·failure·취소·terminal·미관측은 mask한다. 이전 partial 요청에서도 이미
완료된 iteration이 의무를 충족하면 새 양의 목표를 거절한다. 이 결과는 선언한
prior snapshot에 대한 상대적 coverage이며 global novelty·최적 task·일반적인
V 품질이 아니다. 기존 legacy rank·WDL 목표를 변경하지 않는다.

root46의 CPU 검사 **35개(semantic 19·utility 16)**가 통과했다. 이 fixture의
등록·Rust receipt·launch 관측은 합성이며 실제 child 실행 증거와 구분한다.
실제 root47에서는 다음을 별도로 확인했다.

- build44는 Rust 소스 `ce93e4e`, `default+search-work-receipts`의 debug 바이너리를
  등록했다. 빌드 전후와 실행 전 source bytes를 대조했고 바이너리는 38,259,808
  bytes·SHA-256 `bfccd4932f8d80613fe495a1c0e7c1ee3041386cf293e88c7de20560e79d83eb`다.
- 실제 current P parent의 `position_command`와 state/history를 사용한 Rust
  Rules-only receipt는 CPU 검사 0개이며 strict semantic admission을 통과했다.
  stdin 전송 전 `/proc/PID/exe`를 읽어 실제 loaded image를 독립 대조했다.
- 전체 V를 보존한 untrained checkpoint를 CPU FP32로 reload하고 parameter
  observation을 forward 전에 고정했다. 1,394개 private 의미 토큰의 실제 V
  logits도 CPU 결과가 나오기 전에 기록했다. 이후 같은 parent/profile에서
  ResumeTask는 depth 1 baseline·depth 2 after를 완료했고 reused depth 1을
  보고했다. Defer는 정상 no-check였다. 선언한 prior snapshot은 비어 있으며
  앞선 candidate-only 실행을 whole-root prior로 사용하지 않았다.
- strict utility factory와 실제 semantic V forward의 별도 softplus loss는
  **0.5029387474**였다. full V parameter digest
  `eedfe35b2750147ff06c8381b5f6b1c4bd0a6ad7299d945cf02ff2b38bd5775e`는 전후
  동일하다. backward·optimizer·parameter update·GPU·제품 V는 실행하지 않았다.
- 소유 child의 pipe EOF·reap·process group 소멸과 managed scratch 정리까지
  exit 0이었다. CPU affinity 0/2·memory.high 6GiB/max 12GiB/swap 0의 실행 전
  적용을 확인했다. 종료 후 service가 표시한 1.5MiB는 Torch 작업 peak의 근거로
  사용하지 않으며 이번 실행의 정확한 RSS/cgroup peak·VRAM peak는 unknown이다.

논리 근거는 `runs/pals/continuation-cpu-check-03/semantic-model-tests-46.log`,
`runs/pals/actual-semantic-utility-cpu-47/`의 사전 선언·원시 영수증·독립 관측·
핀·인수 결과에 보존했다. `root-result.json`의 SHA-256은
`38b1f38e6303bfdd5788a297556d2578e7509f6ca4df4a0eeede9ab027cae698`,
`frozen-semantic-utility-preparation.json`은
`58c6a8b11c1d471777542be348b31aa0341ff826aaf3c2b4180b48798fa55313`다.
root47의 Python 검증 구간 12,918ms와 설치·관리 정리를 포함한 service 구간은
서로 다른 범위이며 성능 계측 또는 학습 성과로 사용하지 않는다.

`ce93e4e`의 [CI 37683887244](https://github.com/daejunnom/RoveZero/actions/runs/37683887244)는
Linux·Windows·CPU bindings·model CPU 네 job 모두 실제 성공했다. 새 Python
source `34dc701`의 [CI 37685960703](https://github.com/daejunnom/RoveZero/actions/runs/37685960703)도
같은 네 job의 실제 성공을 별도로 확인했다. 서로 다른 SHA의 성공을 재사용한
기록이 아니다.
DG02의 실제 divergence/repair 인과 admission·목표 준비, 외부 helper의 완전한
paired 인수와 전체 학습 준비의 통합은 남아 있다. GPU 검증은 계속 보류한다.

### Native 이탈 지점 입력의 별도 인수와 예약 순서

`ec32bb2`는 실제 native Divergence 입력과 별도의
`rz-pals-native-divergence-context/1` descriptor를 물리 submit 전에 봉인한다.
각 slot의 원래 순서·proposal·상대 차례의 prefix Rules state/history·원래 tensor
sidecar·revision을 연결한다. 다시 준비하면서 샘플한 남은 시간을 원래 입력에
덮어쓰지 않으며, descriptor에서 lineage와 false-learning journal로 이어지는
연결에는 hash cycle이 없다. 과거 collection에 이 descriptor가 있었다고 소급하지 않는다.

독립 소스 검토에서 producer journal 뒤에 output credit을 예약하는 순서 결함을
발견했다. `638858f`는 정확한 출력 바이트와 raw/stage 예약을 producer capture
이전으로 옮긴다. 예산 부족이면 journal·trace·sequence가 바뀌지 않고, 명시적으로
credit을 늘린 뒤 같은 미제출 request를 다시 인수할 수 있다. producer 자체 quota
거절은 이미 예약한 정확한 준비 행과 실패 journal을 보존한다. root50 native CPU
observer 검사 10개, Arena all-target/all-feature Clippy `-D warnings`와 workspace
format이 통과했고 managed scratch 정리도 verified였다. root49의 같은 10개 검사를
더해 고유 검사 수로 표시하지 않는다.

`5851659`의 `native_divergence.py`는 일반 strict dataset selector를 유지하면서
별도 auxiliary input·sidecar·lineage·false journal·source·physical event/raw output을
인수하는 접점이다. `NativeCollectionFacts`는 collection의 읽기 전용 사실이며 개별
입력이나 label의 권한을 주지 않는다. 실제 aux 입력의 public records·role·revision은
같은 root의 ordinary parent와 다를 수 있으므로 원래 captured tensor를 사용한다.
empty candidates는 한 padding과 false input mask로 보존한다. 감독 policy·WDL·task·
divergence mask는 모두 false이며 ranking·수선 성공·학습 정답을 만들지 않는다.
root51의 합성 CPU fixture 19개가 통과했고 실제 모델·Rust child의 종단 성공과 구분한다.

root52는 `638858f`의 정확한 tracked Rust/Cargo 입력을 빌드 전후 대조하고
CPU semantic CLI를 별도 등록했다. compiler artifact는 `fresh=true`, 바이너리는
38,259,808 bytes·SHA-256
`bfccd4932f8d80613fe495a1c0e7c1ee3041386cf293e88c7de20560e79d83eb`로 이전 root44와
같다. 같은 bytes라는 사실과 새 source 등록·빌드 실행을 각각 기록한다.

실제 root53은 retained collection23의 false-learning Divergence 한 건에 대해
새 Rust prefix Rules 영수증·loaded executable inode·stdin 이전 관측·reap·pipe EOF·
owned group 소멸을 확인했다. C forward 전에 신규 Python graph route 검사에서
거절됐다. 추적 결과 이 검사는 raw Rules semantic digest를 composite encoding
digest와 직접 비교했고 export 당시 source와 capture 당시 source도 같은 artifact로
비교했다. collector의 `load_pinned`는 기존 `verify_pals_rules_profile`을 이미 호출하므로
이 실패를 collector의 의미 검사 우회로 해석하지 않는다. 원래 root53 실패·원시 자료와
managed cleanup 성공을 보존하고 Python gate를 실제 계약에 맞게 수정한다.

계약상 capture encoding은
`SHA256(UTF-8("rovezero.pals-board-records.v1") || raw Rules semantic digest 32B)`다.
export의 `rules_encoder_source_sha256`은 export 당시 선언 provenance이고, 실제
capture encoder source는 registry·loaded source·sidecar의 별도 provenance다.
각각 독립 pin을 검증하며 다른 source 시점 자체를 입력 의미 불일치로 판정하지 않는다.
실제 다른 의미·미지원 profile·altered raw asset은 계속 거절해야 한다.

root50·51·52·53의 논리 자료는 `runs/pals/continuation-cpu-check-03/`와
`runs/pals/actual-native-divergence-cpu-53/`에 보존했다. root53은 CPU search·C forward·
학습을 실행하지 않았으며 Rules-only child의 정상 종료가 auxiliary admission 성공을
뜻하지 않는다. 정확한 Torch RSS/cgroup peak와 VRAM peak는 unknown으로 유지한다.
`638858f`의 [CI 37689767011](https://github.com/daejunnom/RoveZero/actions/runs/37689767011)는
Linux·Windows·CPU bindings·model CPU 네 job 모두 실제 성공했다. 후속 Python SHA와
GPU·실제 학습의 인수에 자동 재사용하지 않는다.

후속 `ab7bdb5`는 이 namespace 비교를 바로잡는다. root54의 CPU fixture 20개는
서로 다른 raw/composite digest·export/capture source를 정상 인수하고, 실제 의미·
loaded adapter·loaded encoding 불일치는 거절했다. root51의 19개와 겹치는 suite이므로
합산하지 않는다. 같은 수정 freeze에 대한 독립 소스 재검토에서도 추가 must-fix가 없었다.

실제 root55는 원래 collection23의 auxiliary
`9e6c08d9467b9c330a0a41054be9da17a892281bc24acf4b2405f13cbcd6e22d`를 원래 current
root parent와 연결했다. 새 Rust prefix Rules receipt 1개와 원래 false journal·tensor·
source·export·physical native output의 독립 pin을 소비해 별도 admission을 통과했다.
`derived_from_legacy_prepared`, `auxiliary_has_current_label=false`,
`producer_journal_learning_input=false`를 유지한다. 원래 collector의 launch 사실은
보존된 실제 실행 결과·당시 launcher source로 audit한 caller 관측이며 새로운
collector launch나 과거 before-dispatch descriptor의 증거로 바꾸지 않았다.

CPU FP32로 기존 full untrained checkpoint를 reload·동결하고 원래 captured public
record·query·8-feature를 C 전방 계산에 사용했다. 원래 native FP32 raw 출력과의
최대 절대 차이는 divergence logits `1.78814e-7`, WDL logits `1.90735e-6`, private
latent `2.86102e-6`이었다. absolute `1e-4`·relative `1e-3` 대조를 통과했다. dummy
candidate는 false input mask로 남았고 실제 policy 비교 후보나 정답이 아니다.
policy·WDL·divergence·task 감독 mask는 모두 false이고 loss는 0이다. 미관측 자료를
0점·무승부 또는 양의 ranking 목표로 바꾼 것이 아니다.

full parameter SHA-256
`eedfe35b2750147ff06c8381b5f6b1c4bd0a6ad7299d945cf02ff2b38bd5775e`는 전후 같다.
새 CPU search는 0개이며 실제 모델·Rules 기능 검증 구간은 6,474ms였다. 설치·정리와
다른 fixture를 포함한 service 전체 시간과 구분하고 성능 향상 수치로 사용하지 않는다.
새 Rust child는 원래 deadline 안에서 pipe EOF·reap·group 소멸을 확인했고 managed
scratch 정리도 verified였다. GPU·optimizer 생성·backward·학습 update는 없었다.
Torch RSS/cgroup peak와 VRAM peak는 unknown이다. 논리 자료는
`runs/pals/actual-native-divergence-cpu-55/`이며 `root-result.json` SHA-256은
`2c1e03fe084eaf2c4080931f1ec697b09d2fd227c168bd57a69d1acf07f034bc`다.
새 descriptor의 실제 fresh collection 소비·C 이탈 ranking·P 수선의 인과/비교 목표·
완전한 external helper pair는 이 legacy-input 대조만으로 완료되지 않는다.

소스 `ab7bdb5`의 [CI 37692482991](https://github.com/daejunnom/RoveZero/actions/runs/37692482991)는
Linux·Windows·CPU bindings·model CPU 네 job 모두 실제 성공했다. 마지막 Windows job은
2026-10-07 21:59:30 UTC에 완료됐다. 다음 Repair 구현의 검사와 별개다.

### 최초 Repair 입력의 인과 연결과 후보 비교 준비

`repair_context.py`는 strict ordinary dataset에 이미 인수된 current native Propose
root와 최초 Repair decision의 별도 immutable capability를 추가한다. 원래 capture의
source·export·launch·prepared journal·입력·sidecar·lineage·물리 출력·탐색 소비를
독립 pin으로 검사한다. 같은 producer 요구는 이 root→Repair 인과 연결에만 적용하며
game 전체에 같은 모델을 강제하지 않는다. auxiliary false-learning 입력이나 물리
완료만 된 응답은 이 current Repair 입력을 대신할 수 없다.

두 Rust Rules-only capability는 root proposal의 합법성과 counterexample prefix가
실제 Repair 상태·이력·FEN·차례·보드·합법 수 순서에 도달하는지를 각각 확인한다.
첫 상대 응답은 원래 proposal의 같은 ply와 달라야 한다. Python에서 체스 Rules를
재구현하지 않으며 원시 CPU 점수·native prediction·`search_consumed`를 수선 성공의
증명으로 사용하지 않는다. Context provenance는 `checked_existing_prepared_lineage`다.
후속 `CheckedComparativePairs`는 같은 parent의 정확한 Repair input SHA·row index·
proposer role에만 연결하고, 기존 next-move ordinal 관점·sign·partial mask를 유지한다.

root56의 집중 CPU 검사 18개는 15 PASS / 3 errors였다. 새 fixture가 Repair 후보를
바꾸면서 overlay restriction에 이전 후보를 남긴 것이 원인이었다. 원래 실패 자료를
보존하고 fixture만 맞췄으며 기존 comparative validator를 완화하지 않았다. 별도
독립 검토에서 forward 후 frozen 모델의 eval·requires-grad 플래그를 재확인하지
않는 누락을 발견했다. 전후 공통 `frozen_digest()`로 module eval·CPU/CUDA autocast
비활성·requires-grad false·gradient 부재·유한한 CPU FP32 bytes를 다시 검사한다.
같은 parameter bytes를 유지하며 `train()` 또는 `requires_grad_(True)`로 바꾸는
두 음성 사례를 추가했다.

수정 후 root57의 CPU fixture 20개가 모두 통과했다. 최종 source SHA-256은
`96ad2df260918de323637a28673e7ba8e1e79c261243ddb9d1cbf9608ac56829`, test SHA-256은
`a3b55668c229d3e735db559d1a8fbecb7247327759c2f6657a154c4e5a955dc8`다. 최종 freeze의
독립 재검토에서 추가 must-fix는 없었다. CPU 0/2·memory.high 6GiB/max 12GiB·swap 0·
pids 128을 실행 전에 확인했고 managed scratch의 종료·제거가 verified였다.
root56의 18개와 root57의 20개는 겹치는 suite이며 고유 검사 수로 합산하지 않는다.

이 fixture는 실제 strict·semantic·candidate factory를 소비하지만 source·launch·
Rules·native 출력은 합성 자료다. 실제 fresh child/모델/수집의 성공으로 보고하지 않는다.
양의 frozen ordinal loss도 strategic repair validity·counterexample validity·WDL·
divergence ranking·학습·기력 인수와 구분한다. 원시 자료는 관리 루트의
`runs/pals/continuation-cpu-check-03/repair-model-tests-56.log`와
`repair-model-tests-57.log`에 보존하며 GPU 검증은 계속 보류한다.

### 새 before-dispatch descriptor의 실제 CPU 소비

root58은 `2840283`의 정확한 Rust/Cargo source를 빌드 전후 대조해 CPU native 수집기를
별도 등록했다. debug·`pals-collection-onnx`·locked/offline 바이너리는 116,810,200 bytes·
SHA-256 `2d9fc6edc61dcd4fc0b99ace5ff4ba8cb230b9c538a41865274fc4b283a52fc3`다.
root59는 같은 registered untrained checkpoint·shared P/C export·CPU ORT에서 1게임·
최대 2ply·line 2ply·rounds 2·role 상한 16·전체 30초로 새 수집을 실행했다.
ordinary current P/C 12행과 before-dispatch 이탈 descriptor 4행을 회수했다.
NN 입력 32개(public 16/role 16)·물리 role 완료 16개·탐색 소비 16개이며 물리 종료·
buffer 해제·in-flight 0·process reap/pipe EOF/owned group 종료를 확인했다.
ply-limit 결과는 unknown이며 대국 승패나 strategic 목표로 바꾸지 않았다.

root60의 새 소비 검사는 첫 descriptor를 인수한 뒤 둘째에서 public source 행의
중복을 모호한 출처로 거절했다. C forward는 시작하지 않았고 실패·정리 기록을
그대로 보존했다. 실제 collector는 search마다 `raw_sources`를 초기화하고 같은
global JSONL에 append하므로 동일 public observation의 정확한 행이 반복될 수 있다.
`_pinned_public_row()`는 observation SHA가 일치하는 byte-identical 행만 묶는다.
다른 bytes·revision·record ID로 원래 pin을 대체하지 않고 요청·journal·context·
physical output의 유일성 검증도 유지한다. Repair 소비자는 같은 helper를 사용한다.

root61은 D 23개·Repair 20개의 CPU fixture를 통과했다. 이전 suite와 겹치므로 고유
표본으로 합산하지 않는다. 이어 root62는 실제 새 descriptor 4건을 각각 current root
anchor·original tensor/lineage/false journal·물리 출력에 연결해 인수했다. provenance는
`captured_before_dispatch`이며 과거 derived 입력이나 ordinary current label과 다르다.
기존 full checkpoint를 reload·freeze한 CPU C batch 4의 native 출력 대조 최대 절대
차이는 divergence `2.38419e-7`, WDL `2.86102e-6`, latent `3.81470e-6`으로 기존 absolute
`1e-4`·relative `1e-3` 허용 오차를 만족했다. full parameter SHA는 전후 동일한
`eedfe35b2750147ff06c8381b5f6b1c4bd0a6ad7299d945cf02ff2b38bd5775e`다. 감독 mask는
모두 false이고 loss는 0이며 ranking·수선 유효성·학습의 증거로 사용하지 않는다.

원시 자료는 `runs/pals/strict-native-divergence-capture-cpu-59/`, 실패한
`actual-native-context-cpu-60/`, 후속 `actual-native-context-cpu-62/`와 continuation의
root58~62 실행 파일·로그에 보존했다. root62 `root-result.json` SHA-256은
`c94e0aa6f251402c0ebe70cf7e517785f29b2cfe3ebd7ee59adcae6c9ec1134b`다. CPU 자원은
0/2·high 6GiB/max 12GiB·swap 0·pids 128로 실행 전에 확인했고 managed temporary
환경 제거도 verified였다. CLI는 caller-pinned path와 소유 process group을 관측했으며
loaded executable inode를 stdin 이전에 확인했다고 표현하지 않는다. 미관측 memory/VRAM
peak는 unknown이다. GPU·backward·optimizer·학습은 실행하지 않았다.

Repair source `d891af4`의 [CI 37694038457](https://github.com/daejunnom/RoveZero/actions/runs/37694038457)는
Linux·Windows·CPU bindings·model CPU 네 job 모두 성공했다. 위 public source lookup
수정과 다음 Rust continuation은 각 새 SHA의 검사로 별도 관리한다.

### 전체 ordered line의 Rules·CPU continuation 접점

[`pals_cpu_task/continuation.rs`](../../crates/rz-uci/src/pals_cpu_task/continuation.rs)와
배타적 `--line-continuation` CLI는 `rz-pals-owned-cpu-line-continuation/1`을 추가한다.
기존 V task·candidate-only·semantic schema의 의미와 default mode를 유지한다.
최대 64ply의 전체 ordered line을 Rules로 재생하며 합법적인 반복 move token을 보존한다.
각 ply의 terminal·취소·원래 deadline을 확인하고 terminal 뒤 추가 move를 거절한다.
root와 endpoint는 FEN·정확한 state·전체 known history·완전한 합법 수 순서·차례를
각각 대조한다. line seal은 root state/history와 전체 순서를 포함한다.

ongoing endpoint에서 fresh TT의 `Independent` bootstrap CPU를 한 번만 호출한다.
실제 nodes·quiescence nodes·TT hits·completed depth·completion·score scope·PV를
보존하고 endpoint에서 PV를 Rules로 재검증한다. raw score 관점은
`endpoint_side_to_move`이며 root 부호나 line 간 전략적 rank로 자동 투영하지 않는다.
terminal endpoint는 정확한 Rules descriptor와 CPU 0회·`report=null`을 반환한다.
`training_target_created=false`, `product_verifier_enabled=false`를 유지한다.
이 접점은 whole-line 합법성과 정확한 endpoint 검사의 사실이며 수선 성공·C 이탈
ranking·WDL 목표·제품 V 활성화의 근거가 아니다.

원래 CLI 시작 Instant의 deadline을 ingress·등록 실행 파일 확인·Rules 준비·CPU·
report 검증·직렬화·bounded output에 이어 사용한다. search reserve는 원래 allowance에서
차감한다. 실패는 실제 work/report/receipt를 가능한 범위에서 typed diagnostic으로
보존하고 비관측 counter는 null이다. 이미 만료된 admission이나 delivery에 새 stderr
grace를 주지 않는다. dispatcher 인수와 CLI self-image·외부 caller 등록/실행 증거는
서로 다른 authority다.

root65는 새 module 9개·CLI 전체 10개(신규 3개/기존 7개), `rz-uci` all-target/all-feature
Clippy `-D warnings`, workspace format 검사를 통과했다. 독립 source freeze 재검토에서
추가 must-fix는 없었다. managed scratch 종료·정리도 verified였다. CPU default build의
기존 feature별 dead-code 경고는 all-feature Clippy 성공과 구분한다. 같은 검사를 고유
표본으로 중복 합산하지 않는다. 원시 로그는 continuation 관리 경로의
`continuation-lib-65.log`, `continuation-cli-65.log`, `continuation-clippy-65.log`와
`continuation-format-65.log`다.

이 절의 초기 확인은 library·CLI correctness 범위다. 이후 새 source의 실제 child와
durable caller anchor를 인수한 root67은 아래에 별도로 기록한다. Repair next-move의
root64 인수와 C divergence ranking·외부 helper paired 종단은 서로 다른 범위다.
GPU 검증과 실제 학습은 이번 실행에서 계속 제외한다.

### 실제 최초 Repair의 causal·ordinal CPU 소비

root63은 등록된 root58 수집기를 재사용하되 line 3ply·rounds 1·최대 1ply 게임으로
새 실행 조건과 source 등록을 고정했다. 기존 line 2ply에서 opponent reply 뒤의 prefix가
line 상한에 도달하여 실제 Repair 모델 호출이 생기지 않았던 경로와 구분한다.
ordinary 7행·native 역할 호출 8건(Propose 3/Divergence 1/Reply 2/Repair 2), NN 입력
16개(public 8/role 8)를 회수했다. 물리 역할 완료·탐색 소비는 각각 8건이며 정상 shutdown·
buffer 해제·in-flight 0·process reap/EOF/group 부재를 확인했다. ply-limit은 unknown이다.
수집기 loaded inode를 stdin 이전에 관측했다고 소급하지 않으며 실제 caller path/group
증거를 사용한다. 이 실행은 새 source를 빌드했다는 주장이 아니라 root58의 독립 build
등록·immutable binary를 재사용한 실제 collection이다.

root64는 outcome 이전에 첫 Repair `request_sequence=6`, proposal `[1609,3063,1487]`,
counterexample `[1609,2745,1032]`, prefix `[1609,2745]`와 Rules 순서의 첫·마지막 합법
후보를 고정했다. historical root52/source `638858f`의 등록 binary에서 Rules-only child
2개로 정확한 root/proposal/prefix/tail을 확인했고 CPU 탐색은 0회였다. 별도 fresh candidate
child 2개는 H2·N100000·TT16·q4·각 10초의 같은 조건으로 실행했다. binary loaded inode를
stdin 전에 대조하고 exit 0·reap·EOF·owned group 부재를 각각 확인했다.

persisted bank의 실제 bytes를 독립 pin한 뒤 strict reload가 반환한 새 parent 객체에
동일 causal bytes를 재인수했다. 이 재연결을 Rules child 재실행으로 세지 않는다.
Repair 입력 SHA `0aeb5caaf2cf03a39b9091d9f0639e9b7c4225a8544116ba07209c65f3ce0e52`의
활성 ordinal pair 1개·sign -1과 frozen softplus loss `1.1267693042755127`을 확인했다.
full V를 보존한 checkpoint `b07a2af…`의 parameter SHA는 전후 동일한 `eedfe35…`였다.
optimizer 생성·backward·update·GPU는 0이다. 이 목표는 다음 Repair 수의 조건부 유한 깊이
순위이며 strategic repair validity·counterexample validity·counterfactual WDL·C divergence
ranking을 인수하지 않는다.

driver 검토에서 all-masked 분기에도 dispatch/identity/cleanup 실패를 먼저 거절하도록
공통 검사를 강화했다. 정상 partial/tie만 HOLD이며 실패 producer를 HOLD로 숨기지 않는다.
final artifact fsync와 stdout 뒤에도 원래 180초를 확인한다. driver의 사전 publication 시간과
supervisor의 실제 exit/pipe/cleanup 시간은 별도로 기록했다. 후자는 19.657초·exit 0이며
copied CPU 환경 설치를 포함한 service는 59.630초였다. 성능 비교 자료로 사용하지 않는다.
managed temporary 제거·tree cleanup은 verified이고 실제 peak는 unknown으로 둔다.

원시 자료는 `runs/pals/strict-native-divergence-capture-cpu-63/`,
`actual-native-repair-cpu-64/`와 continuation의 root64 supervisor 기록에 보존한다.
root64 `root-result.json` SHA-256은
`0fa09d08eeefffdd419ac13ad314bf11244a3f64b884b3c5c6c422541e226bde`다.
root64의 CPU checker source 재사용을 다음 root66의 새 Rust continuation 인수로 대체하지 않는다.

### 새 continuation binary 등록과 정확한 CI

root66은 `83b886c4bd65e738bcf0738ecff7265081313fbb`의 Git-tracked Rust/Cargo 230개를
빌드 전후 대조했다. immutable CPU 바이너리는 39,208,872 bytes·SHA-256
`6c6ec1ebf8ab12353059a104ee3760f30adbbdc228a8f8769a70d6feb9906ed6`이며 debug·
default+search-work-receipts·locked/offline·Rust 1.96.0을 기록했다. Cargo의 compiler artifact
`fresh=false`도 보존한다. build exit 0·source 전후 일치·managed cleanup을 확인했으며 이
등록만으로 실제 새 continuation child나 비교 소비를 인수하지 않는다.

public source lookup 수정 `c8f22cb`의 [CI 37695229359](https://github.com/daejunnom/RoveZero/actions/runs/37695229359)와
Rust continuation source `83b886c`의 [CI 37696418278](https://github.com/daejunnom/RoveZero/actions/runs/37696418278)는
Linux·Windows·CPU bindings·model CPU 네 job 모두 실제 성공했다. 후자의 Windows job은
2026-10-07 22:34:48 UTC에 완료했다. 이전 진행 중 관측과 최종 성공 관측을 구분하고,
GPU·전체 paired 인수·후속 source의 검사를 대신하지 않는다.

### 등록된 전체 수순 continuation의 실제 CPU 인수

root67은 root66의 immutable binary와 Rust/Cargo 230개 source 등록을 재사용했다.
actual63의 첫 Repair에서 outcome 이전에 선택한 proposal·counterexample 두 3-ply
수순과 같은 startpos의 사전 지정 Fool's Mate 수순을 검사했다. Rules-only child 3개를
모두 마친 뒤 각각의 정확한 root·endpoint·전체 known history·합법 수 순서를 caller의
durable anchor에 고정하고 continuation child 3개를 실행했다. 두 단계는 같은 자체
Rules 구현을 사용하며 독립 규칙 oracle 대조라고 표현하지 않는다.

ongoing 두 endpoint는 모두 black-to-move이며 fresh H2·N100000·TT16·q4 CPU를 각각
1회 호출했다. 완료 깊이 2·depth_limit·completed iteration의 raw endpoint 점수는
proposal 0, counterexample 8이었다. nodes는 각각 137/278, quiescence nodes는
116/253, TT hits는 0/1이다. 순서가 고정된 전체 3ply와 정확한 endpoint에서 관측한
raw 값이며 root 관점 순위·전략적 반박·수선 성공·WDL 목표로 자동 변환하지 않는다.
terminal 수순은 black winner의 Checkmate·white-to-move·CPU 0회·report null을 확인했다.

6개 Rust child 모두 실제 loaded executable inode를 stdin 전달 전에 대조했고 exit 0·
reap·stdout/stderr EOF·owned group 부재·원래 caller deadline을 확인했다. supervisor의
실제 driver exit·pipe·group 정리까지는 14.267초, managed service는 17.854초였다.
모델·Torch import·optimizer·backward·GPU·target 생성은 0이며 성능 비교가 아니다.
CPU affinity 0/2·high 6GiB/max 12GiB·swap 0·pids 128, managed temporary 제거와
tree cleanup을 확인했다. 종료 후 service가 보고한 작은 peak는 실행 peak 근거로
채택하지 않으며 실제 memory/VRAM peak는 unknown이다.

원시 자료는 `runs/pals/actual-line-continuation-cpu-67/`와 continuation의 root67
supervisor 기록에 보존한다. `root-result.json` SHA-256은
`fcdfa0e500259a9c6ff9008125eaea217a07077f3c17f6969b493a9317c9ebfe`다.
원래 factual receipt의 before-result anchor는 ordinal criterion이 아니므로 후속
ordinal consumer에 소급하여 인수하지 않는다.

실행 전 독립 검토에서는 기존 helper55의 child 생성 이후 selector 초기화 실패가
cleanup 밖에 있던 경로를 발견했다. 과거 원시 자료와 helper55는 보존하고 새 helper67을
별도 pin했다. selector·buffer를 spawn 전에 준비하고 모든 post-spawn 설정을 같은
try/finally로 보호했다. root70의 오류 주입 6개는 no-spawn·Popen failure·실제 child의
register failure·selector close diagnostic·wrong image·timeout·aggregate overflow를
검사했고 모두 통과했다. 실제 child의 signal·reap·group 부재·pipe 정리를 포함하지만
이 outside-Git 실행 보조의 성공을 제품 arena 전체 오류 처리 인수로 확대하지 않는다.

### 이탈 후보 수집의 관측 범위와 후속 순위 준비

root68/69는 root58/source `2840283`의 등록 CPU native collector를 재사용한 별도
유한 수집이다. 같은 untrained 모델·CPU ORT에서 최대 1ply·line5·rounds2·beam1을
고정했고 root68은 per-go node 1024/total 8192, root69는 결과 전에 새 질문으로
per-go 32768/total 32768을 선언했다. 원래 실패·제한 자료의 예산을 소급 변경하지 않았다.

root68은 ordinary 9행·auxiliary context 1건, root69는 ordinary 26행·context 2건을
회수했다. 각 proposal에는 opponent-to-move slot 2개가 있으나 현재 제품 탐색은
round마다 slot 하나를 선택한다. 다음 round에서는 public revision과 proposal이
바뀔 수 있으므로 다른 context의 Repair를 첫 D 입력의 두 번째 slot 증거로 묶지 않는다.
이는 beam 폭의 의미나 단순 node 예산 부족과 구별한다. 현재 자료에는 같은 exact
auxiliary 입력에서 비교 가능한 slot 두 개의 실제 Repair 결과가 없으며 C pointer의
known ranking pair는 아직 만들지 않았다. 미검사 slot은 not-examined로 유지한다.

두 수집의 process exit 0·reap·EOF·owned group 부재·native 물리 종료·buffer 해제·
in-flight 0과 managed temporary 정리를 확인했다. collector의 loaded inode를 stdin
전에 관측했다는 소급 주장은 하지 않는다. ply-limit 결과는 unknown이며 target·학습·
기력으로 승격하지 않는다. 원시 자료는 `strict-native-divergence-capture-cpu-68/`,
`strict-native-divergence-capture-cpu-69/` 관리 경로에 보존한다.

문서 커밋 `866f5b2`의 [CI 37697648405](https://github.com/daejunnom/RoveZero/actions/runs/37697648405)는
Linux·Windows·CPU bindings·model CPU 네 job 모두 실제 성공했다. root67/68/69의
실제 실행 source·기능 인수와 CI 커밋은 각각 구분하며 GPU 검증은 계속 보류한다.

### 전체 수순의 사전 criterion과 조건부 ordinal consumer

[`whole_line_ordinal.py`](../../experiments/model-research/pals/src/rz_pals_model/whole_line_ordinal.py)는
기존 strict current selector와 D/Repair capability를 재사용하고 새 의존성·공통 revision·
제품 wire를 바꾸지 않는다. raw endpoint 값 자체는 그대로 두고 동일 L/H/N/Q/TT/wall·
profile·전체 known history의 두 forced line을 결과 전 `line_conditioned_surrogate`
criterion으로 비교한다. criterion의 명시 방향과 parity만 root surrogate에 적용하며
unrestricted endpoint CPU 완료를 whole-line minimax 완료로 표시하지 않는다.

factory는 7개 asset의 실제 bytes와 각 task의 request/receipt/stderr/launch/process
bytes 및 독립 pin을 요구한다. historical build 재사용은 actual build exit·source 전후
manifest·binary의 독립 caller 관측 범위로 명시하며 consumer가 다시 컴파일했다고
표현하지 않는다. before-plan·criterion의 durable-before-spawn과 실제 loaded inode·
exit·reap·EOF·group 부재·원래 allowance·source/binary 안정성을 known 및 masked
모두에서 검사한다. 다른 public revision이나 auxiliary 입력을 현재 ordinary label로
대체하지 않는다. fake callback·metadata-only parent·self-pinned bytes는 authority가 아니다.

terminal·mate band·range·tie/margin·partial·canceled·missing·unknown full history는
raw 관측과 별도 mask reason으로 남는다. identity·namespace·resource·cleanup 실패는
ValueError로 거절한다. 반복 이력이 완전해졌어도 FEN 이전 전체 이력 불명을 known으로
바꾸지 않는다. capability는 role projection이 필요한 조건부 비교이며 ordinary policy·
C learning label·Repair validity·counterexample validity·WDL·minimax·전술 증명·학습
target 생성 권한은 false다. C slot 및 Repair 첫 결정으로의 projection/loss는 후속 단계다.

root73의 CPU synthetic contract fixture 23개는 통과했다. source/test pin은 실행 전후
동일했고 CPU Torch 2.8.0+NumPy 2.2.6·affinity 0/2·high 6GiB/max 12GiB·swap 0·pids
128을 확인했다. managed copied 환경의 tree cleanup·temporary 제거는 verified다.
fixture의 Rules/build/launch 사실은 합성이므로 실제 child·양의 native 목표·모델 loss·
학습·GPU의 성공으로 해석하지 않는다. 이 consumer를 실제 새 before-criterion child와
연결하는 root74는 아직 미실행이며 root67 사실을 소급 인수하지 않는다.
원시 로그는 continuation 관리 경로의 `whole-line-model-tests-73.log`,
`whole-line-model-supervisor-73.log`에 보존한다.

독립 검토는 별도 selector close 오류가 `transport_failure`와 달리 남을 수 있는데
초안의 process 관측 schema에 그 축이 없던 점을 지적했다. 새 `cleanup_error` 필드를
필수로 보존하고 known·allmasked 모두 None을 요구한다. 이번 owned Rust CLI는 정상
typed stdout와 빈 stderr만 인수하며, 실제 nonempty stderr를 정상 receipt나 mask로
숨기지 않는다. root75는 해당 두 경계의 known/allmasked 오류 주입을 포함한 25개
CPU fixture를 통과했다. source/test 전후 pin 일치·managed temporary 및 tree 정리를
확인했다. root73과 겹치는 suite이므로 23+25개 고유 검사로 합산하지 않는다.
새 source SHA-256은 `82bdd8d550fbae1bb332ea0d6e27f61745f8e7e372b33ae25d4ffa61cb94bc01`,
test는 `4652166797a046c6836bc38227917fe94835431e8ffe7fd05b643f21cb9f3327`이다.
원시 로그는 같은 관리 경로의 `whole-line-model-tests-75.log`와 supervisor 기록에
보존하며 실제 before-criterion root74 인수와 구분한다.

### 새 사전 criterion의 실제 whole-line 인수

root74는 이전 root67 factual 영수증을 읽지 않고 새 Rust child를 실행했다. 등록된
root66 binary의 실제 capabilities child 1회(CPU 0) 뒤, actual63의 strict current
Propose root와 첫 Repair의 두 3ply line을 독립 원시 source pin으로 연결했다. 새
criterion·selection을 durable 게시하고 Rules-only child 2회(CPU 0)로 endpoint를
준비한 뒤 최종 before-plan을 저장하고 fresh continuation child 2회(CPU 2)를 실행했다.

`maximize_root_surrogate`·L3·H2·N100000·TT16·q4·margin1·score limit20000을 사전에
고정했다. 두 black endpoint의 raw 값은 0/8, criterion의 root surrogate는 0/-8이며
조건부 known pair 1개·sign1을 인수했다. 원시 점수의 의미는 유지했고 강제 전체 line의
유한 endpoint 비교를 minimax·반박·Repair 성공으로 바꾸지 않았다. 두 line은 같은
첫 move를 갖기 때문에 이 pair를 다른 Repair next-move 두 후보로 만들지도 않는다.
ordinary policy·C slot·WDL·학습 target·loss·role projection은 생성하지 않았다.

immutable binary는 8MiB 출력 은행 밖의 exact path/pin으로 참조하며 reload 때 실제
bytes를 다시 읽었다. criterion·before-plan·request·receipt·launch·process 관측과
독립 pins를 저장한 뒤 strict frozen parent도 새 객체로 reload하여 같은 factory와
verify 결과를 확인했다. capabilities/Rules/continuation 5개 child 모두 loaded inode를
stdin 전에 확인했고 exit 0·reap·EOF·group 부재·빈 stderr·cleanup error 없음·원래
allowance를 관측했다. historical build66/source `83b886c`를 재사용한 것이며 이번
새 빌드나 현재 Rust source 비교로 표현하지 않는다. 모델 생성·forward·checkpoint
읽기·optimizer·backward·학습·GPU는 0이다.

실제 driver exit·pipe·group 정리까지는 14.128초, CPU 환경 설치를 포함한 service는
54.369초였다. 기능 인수이며 성능 비교가 아니다. 실행 전 affinity 0/2·high 6GiB/max
12GiB·swap 0·pids 128과 managed temporary/tree cleanup을 확인했다. 실제 peak는
unknown이다. 원시 자료는 `runs/pals/actual-whole-line-ordinal-cpu-74/`와 continuation의
root74 supervisor 기록에 보존한다. `root-result.json` SHA-256은
`ad1ada19b4db4406d5d4d9176ae86ac2fa49f456e575b51001e2566bc1b945d8`다.

### 제품 CPU 등록과 외부 helper의 동일 PID 기능 검사

root72는 별도 debug·CPU ONNX feature·locked/offline 빌드에서 Rust/Cargo source 230개를
전후 대조하고 UCI와 main arena 바이너리를 등록했다. Windows Git worktree pointer를
WSL Git이 해석하지 못한 root71 preflight 실패는 Cargo 시작 전 실패로 보존했으며
Git 설정을 수정하지 않았다. 기준 Git SHA는 Windows에서 고정하고 WSL에서 같은
file bytes를 확인했다. 등록 자체는 제품 NN·대국 인수가 아니다. PALS pair 실행의
실제 진입점은 main arena와 다른 `pals_pair_prepare` example이므로 이를 대체하지 않는다.

root76은 기존 Stockfish 19 pin·T2·Hash16MiB profile을 사용한 새 helper CPU03 기능
검사다. 실제 `/proc/PID/exe`를 첫 stdin 전에 확인한 단일 PID에서 startpos 및
`startpos moves e2e4 e7e5`를 차례로 입력하고 각 `go depth 2 nodes 4096 movetime 900`
뒤 ready barrier를 확인했다. 보고된 depth2 nodes는 148/73, bestmove는 e2e4/g1f3이며
stop·마감 도달·재시작·출력 잘림 없이 종료했다. 옵션은 광고 type/range·설정 명령·
readyok로 확인했지만 query 불가능한 실제 적용값까지 관측했다고 표현하지 않는다.

정상 quit exit 0, unreaped leader의 TERM/KILL 뒤 실제 reap·두 pipe EOF·group 부재를
확인했다. 최종 stdout부터 실제 driver exit·EOF·group 정리까지 0.722초였고 최초
60초·합산 output1MiB·line16KiB·stdin4KiB를 지켰다. managed temporary/tree 정리도
verified다. foreign UCI가 보고한 nodes 검사이며 물리 hard-node cap·합법 수·PALS
제품 소비자·paired 강도·RoveZero NN·GPU 인수가 아니다. 실제 memory peak와 이
standalone probe의 inherited affinity는 별도 관측하지 않았으므로 unknown이다.

원시 자료는 `external-helper-pair-cpu-03/attempt-01/`과 continuation의
`stockfish-helper-supervisor-76.json`에 보존했다. 기존 CPU02의 node256 실패와
불완전 paired 결과를 이 성공으로 대체하지 않는다. 새 paired 실행은 제품 UCI와
PALS example의 별도 등록, 새 조건·source·manifest digest 및 종단 인수를 요구한다.

### 최초 Repair 결정의 whole-line projection

[`whole_line_projection.py`](../../experiments/model-research/pals/src/rz_pals_model/whole_line_projection.py)는
별도 `CheckedRepairWholeLineProjection` factory로 정확한 `CheckedRepairContext`와
`CheckedWholeLinePair`를 연결한다. 같은 strict parent 객체·현재 Repair row·동일 causal
anchor 객체와 `native_initial_repair`를 요구하며, continuation root의 전체 Rules
descriptor와 이력을 반례 prefix의 실제 Rules target과 대조한다. 비교하는 line은
현재 Repair 상태에서 시작하는 suffix다. root74의 ordinary Propose root 전체 수순
비교는 이 projection의 실제 양성 증거가 아니다.

결과 없는 별도 before 문서는 causal/current context, 원래 ordinal plan·criterion의
byte pin과 ordered task/first move만 담는다. 결과 whole SHA·score·sign을 넣지 못하며,
두 실제 launch의 pin과 durable-before-spawn 관측을 독립 caller bytes에 결합한다.
`maximize_root_surrogate`에서 서로 다른 합법 첫 수만 별도
`next_repair_move_whole_line_conditioned_surrogate` 목표로 소비한다. 같은 첫 수에 다른
suffix는 `unsupported_same_first_move`로 보존하고 duplicate logit collation을 거절한다.
일반 policy·WDL·전략적 Repair/반례 validity·C ranking·minimax·전술 증명은 부여하지 않는다.

offline frozen helper는 독립 checkpoint reload observation과 현재 checkpoint/epoch,
실제 config·유한 CPU FP32 parameter bytes를 확인한다. caller의 원래 absolute deadline과
forward 전후 eval·requires_grad false·grad 없음·autocast 없음·parameter 및 parent 재검사를
유지한다. all-masked 준비는 forward 전에 거절하며 backward·optimizer·update는 없다.
제품 Rust 엔진과 기존 wire·workspace·공통 revision은 변경하지 않는다.

root78은 CPU Torch 2.8.0/NumPy 2.2.6에서 집중 합성 fixture 27개를 통과했다.
소스와 검사는 각각 `448e39f4733559955fd36a97eeffb19cfbebef9d63f52f1f74e9dca55f80432d`,
`91d03a2ecd1c4f76e53bae76cb3d0ee2701dcc0edb8004be8c1509521caba82d`이며 실행 전후 일치했다.
작은 numeric stand-in의 loss 배선 검사로서 실제 Rust Rules child·checkpoint reload·
조건부 Repair projection의 실제 양성 인수·학습·기력 증거는 아니다. affinity 0/2,
high 6GiB/max 12GiB·swap 0·pids 128과 managed temporary/tree 정리를 확인했다.
service 115.255초에는 환경 설치가 포함되며 성능 비교가 아니다. 종료 후 service가
표시한 peak로 실제 model/system peak를 판정하지 않으며 해당 peak는 unknown이다.
원시 자료는 continuation 관리 경로의 `repair-projection-model-tests-78.log`와 resource,
managed 기록에 보존한다. GPU 검증은 계속 보류한다.

기존 `c483512`의 [CI 37702463443](https://github.com/daejunnom/RoveZero/actions/runs/37702463443)는
네 CPU job 모두 성공했고 Windows job은 2026-10-07 23:35:16 UTC에 완료됐다.
이 결과에 신규 projection 소스의 CI 인수를 포함시키지 않는다.

후속 `8dde262`의 [CI 37705295820](https://github.com/daejunnom/RoveZero/actions/runs/37705295820)은
Linux·Windows·CPU bindings·PALS model CPU 네 job 모두 실제 성공했다. Linux 최종 완료는
2026-10-08 00:05:24 UTC다. 해당 SHA의 CPU 검사이며 새 C witness와 소스 밖 supervisor,
GPU·실제 학습·최종 paired 인수를 포함하지 않는다.

### 조건부 C slot의 실제 관측 연결

[`native_slot_repair.py`](../../experiments/model-research/pals/src/rz_pals_model/native_slot_repair.py)의
`CheckedNativeSlotRepair`는 하나의 exact D 입력·ordered slot과 원래 root, Reply, 최초 Repair,
후속으로 실제 선택된 public repaired line을 전체 관측 창에서 도출한다. caller가 matching
index나 callback을 제공할 수 없으며, root·Reply·초기 Repair·publication은 기존 strict current
view에 있어야 한다. 나머지 historical 호출은 등록된 immutable raw 관측으로만 남긴다.
prepared/input/sidecar/lineage/event/raw-output 전체 roster와 per-search work summary·native
finish를 먼저 대조하므로 빠진 호출을 완전한 미검사 창으로 표현하지 않는다. 원래 collector
build manifest와 binary, 검토한 Rust transition 두 소스 및 실제 Rules capability의 완전한
이력을 각각 요구한다.

기존 wire에는 직접 D→Reply→Repair causal ID가 없으므로 범위는
`conditional_unique_prepared_lineage`다. 일반 flow의 publication은 다음 D에서 처음 관측될
수 있어 이전 window의 `not_examined`를 채우지 못한다. 좁은 ordinary-observer 경로의
synthetic known은 실제 collector의 branch/role-limit 경로를 실행한 증거가 아니다.
Reply 소비를 Reply policy가 CPU 응답을 선택했다는 주장으로 바꾸지 않는다. 전략적 수선,
counterexample validity, C ranking, whole-line ordinal·target·loss·학습·제품 권한은 모두 false다.

독립 검토에서 exact public raw 반복 거절, 같은 root input revision 역행, physical output의
bool request ID가 정수와 같게 비교되는 경계를 수정했다. 동일 observation SHA와 실제 lexical
bytes가 모두 같은 반복만 합친다. 같은 game/revision에서 record_index·critical만 달라진
selected projection은 `unsupported_public_projection_change`로 보존하며 뒤의 immutable
충돌을 끝까지 검사한다. 빈 raw와 빈 selected는 부재이고 selected에 필요한 raw가 없으면
거절한다.

중앙 root82는 30개 모두 공통 semantic fixture 준비에서 ERROR였다. 빈 prefix/root_moves에
부적격 task를 지정한 fixture를 기존 eligibility 조건에 맞추고 제품 validator는 유지했다.
새 root84에서 CPU Torch 2.8.0/NumPy 2.2.6의 합성 fixture 30개가 10.299초에 통과했다.
소스 `84b474a1809cb2a9dc63793ad06c715023e1bb0d2e78ada0ba3073b41902ce32`와
검사 `1cc5273d740eea2f97976fabdea26b17b0ac934da778eb94f7aaa103c34fc2be`의 전후 pin,
affinity 0/2·high 6GiB/max 12GiB·swap 0·pids 128과 managed temporary/tree 정리를 확인했다.
설치 포함 service 60.113초는 성능 비교가 아니며 실제 메모리 peak는 unknown이다.
root82의 원래 30 ERROR와 cleanup 증거는 보존하고 검사 수를 합산하지 않는다.
actual69·actual74를 새 witness나 C 감독으로 승격하지 않았다.

### 최종 CPU helper 대국의 준비와 감독

새 CPU03의 실제 준비 root79는 lock/opening/private snapshot 세 command의 loaded image,
exit 0 및 transport 정리를 확인했다. driver exit·pipe·정리 22.379초, managed temporary/tree
정리 완료이며 engines/NN/paired 인수는 false다. 모델·binary·기존 조건과 새 source freeze
revision 2를 고정했다. PALS(+Stockfish 19 CPU_R) 대 자체 CPU의 120초+1초·흑백 교환 두 판,
동시 대국 1개·최대 256 ply·전체 900초+정리 30초를 실제 실행해야 최종 인수가 가능하다.

소스 밖 supervisor 독립 검토에서 취소가 cleanup을 끊는 경계와 stdout 저장 오류가 stderr
보존을 생략하는 경계를 수정했다. 새 helper는 작업 구간의 cooperative 취소만 검사하며
handler는 flag만 설정한다. 각 stdout/stderr와 nested marker/result의 원시 보존을 따로
시도하고 첫 오류를 유지한다. 바깥 PGID 종료는 별도 nonce systemd service 종료가 아니다.
최종 PGN·clock·typed Core·NN·helper·물리 수명 인수는 transport와 구분한다.

root83은 중앙 실행 사용자 설정이 fixture의 root 전제와 달라 child 없이 실패했다. 원래 자료를
보존하고 evidence namespace만 바꾼 root85를 root로 실행해 stdlib 오류 주입 6항목을
0.432초에 통과했다. 실제 tiny child의 종료·reap·group과 로그 보존을 검사했지만 cleanup
handler 직접 호출을 외부 OS signal 전달 성공으로 보고하지 않는다. root80의 Windows CLI는
WSL app alias의 binary 읽기 단계에서 35ms에 실패했고 engines/NN은 시작하지 않았다.
이 실패를 보존하고 system WSL 실행 파일을 명시한 별도 CLI 관측을 준비한다. 실제 paired
대국 인수는 아직 없으며 GPU·학습·새 cloud 실행은 없다.

### CPU03 실제 대국과 Core 탐색 실패 보존

중앙 root86은 system WSL 실행 파일을 사용해 실제 CPU03 한 쌍을 실행했다.
Windows CLI exit 0과 전체 75.919초, 바깥 transport 68.428초, nonce service의
정상 제어 호출 종료·기존 cgroup 부재를 관측했다. 원시 자료 48파일 387,832,084B를
Windows 관리 루트로 회수해 크기·SHA256를 전부 대조했으며 Linux 원본도 보존했다.

두 게임은 58/33 ply 체크메이트로 끝났고 PALS는 흑백 모두 졌다. 120초+1초 시계의
91개 실제 차감 기록과 흑백 교환, Rules 종료·PGN, native/helper 정상 종료는 일치했다.
그러나 PALS go 45건 중 실패 6건이 있어 전체 기능 인수는 실패다. 외부 checker가
task당 4096을 넘는 4844/6443 노드를 보고한 두 건과, 종료된 checker의 후속 호출
거절 네 건을 원시 work에 보존했다. 마지막 합법 착수로 게임이 계속된 사실은 실패를
없애지 않는다. GPU·학습·기력 향상·전체 메모리 peak는 이 실행의 인수 범위가 아니다.

기존 Core 조립기는 실제 `failed_returns`를 읽고도 `pair_eligible=true`를 생성했다.
원래 Core와 PGN을 재작성하지 않고, 새 코드에서 endpoint의
`search_failed_go_count: Option<u64>`와 typed `SearchFailure`를 연결했다.
과거 V3의 필드 부재는 역직렬화·재직렬화 시 unknown으로 유지한다. 새 OwnCPU/PALS
양성 인수에는 관측된 0이 필요하며, 양수이면 실패 목록과 부적격 상태를 모두 요구한다.
Reference UCI에 RoveZero 내부 계수를 만들지 않는다. V1/V2·공통 revision·모델·탐색·
요청 노드 예산과 PGN의 실제 결과·종료 이유는 변경하지 않는다.

root87의 format 검사 실패를 보존하고 formatter 적용 뒤 root88에서 format,
`rz-experiments` library 81개·`rz-arena`의 `pals-collection-onnx` library 110개를
통과했다. 실제 affinity 0/2·high 6GiB/max 12GiB·swap 0·pids 128을 확인했고,
managed 임시 tree 정리도 완료했다. arena 검사 69.95초와 설치 없는 service 104.014초는
정확성 실행 시간이며 성능 비교가 아니다. 종료 후 표시된 작은 service peak는 전체
메모리 peak의 근거로 쓰지 않는다. 새 자체 Rust checker pilot은 별도 명세·등록 바이너리로
진행하며 CPU03의 외부 checker 실패를 재시도나 성공으로 바꾸지 않는다.

`c7c1d90`의 [CI 37708314220](https://github.com/daejunnom/RoveZero/actions/runs/37708314220)는
네 CPU job 모두 실제 성공했다(Windows 최종 2026-10-08 00:39:30 UTC).
이 결과에 후속 Core 수정의 CI·실제 대국 인수를 포함시키지 않는다.

Core 수정 `0027464`의 Windows job은 새 테스트 helper와 enum import의 cfg 범위가
달라 컴파일 단계에서 실패했다. Linux/test 조건을 동일하게 맞추는 import 수정으로
연결했으며, Linux의 유효 바인딩과 실행 로직은 동일하다. 원래 Windows 실패 로그와
후속 CI는 별도로 보존한다. 역사적 V3 decode를 새 양성 인수와 혼동하지 않는다.

중앙 build89는 `0027464`의 Rust/Cargo 230개를 빌드 전후 대조하고 CPU UCI와
PALS pair example을 등록했다. Cargo exit 0·managed 임시 정리 완료·service 13.100초를
관측했다. UCI는 `fresh=true`이고 기존 72의 동일한 78,293,936B/
`7097c6182172f1ed4313a35242f0ccf949fef05ce743ab56ecc55513db8af84b`다.
Pair example은 `fresh=false`이며 113,055,936B/
`3ef8a76fbe30fb636c94bbdd6816208fca163844e9c714ca7b1684c5fbd0c50c`로 별도 등록했다.
이는 실제 역사 소스 `0027464`의 Linux 바이너리이며 이후 Windows import 수정이나
문서 커밋에서 새로 빌드한 것으로 표시하지 않는다. 새 자체 CPU_R pilot이 이 등록을
사용할 때 모델·시계·자원·source proof와 최종 PR 검사를 각각 기록한다.

### 최종 CPU CI와 추가 실행 보류

Windows import 조건 수정 `f533b97`의
[CI 37710950947](https://github.com/daejunnom/RoveZero/actions/runs/37710950947)는
Linux·Windows·CPU bindings·PALS model CPU 네 job 모두 실제 성공했다.
최종 Windows 완료는 2026-10-08 01:09:52 UTC다. 기존 `0027464`의 Windows
컴파일 실패를 보존하며, 새 CI를 역사적 build89 바이너리의 새 빌드나 실제 대국
성공으로 표시하지 않는다.

별도 actual81 Repair suffix의 실제 CPU 검사를 위한 환경 준비 제어 호출은
exit 1로 종료됐다. 실제 도달 단계는 미확인이며, 두 준비 로그는 0B였고 driver 결과·최종 managed
정리 영수증은 얻지 못했다. 이후 Ubuntu가 중지 상태임을 확인했으나 원인을
OOM으로 확정하지 않는다. unit 부재만으로 소유 lease·scratch의 정리 완료를
인수하지 않으며, 원본 시도와 로그를 보존한 상태에서 같은 namespace를 다시
실행하지 않는다. 모델 실행·frozen loss·checkpoint 인수는 미확인이다.

현재 GPU 검증은 사용자 지시로 계속 보류한다. 추가 CPU 신경망·paired 실행은
호스트 메모리 여유 확인과 중단된 소유 작업의 복구 확인 뒤에만 진행한다.
그동안 별도 CPU04 자체 Rust CPU_R pilot의 소스 준비를 수행한다. 독립 검토에서
검증된 Python 원문과 bytecode cache의 실행 경로 차이, 최종 준비 실패 뒤 성공
영수증을 재사용할 수 있는 경계를 발견했다. 확인된 원문 bytes의 직접 실행과
최종 성공·실제 exit 0의 별도 인수를 연결하기 전에는 대국을 시작하지 않는다.

최소 P6의 자체 CPU 상대 pilot과 외부 Stockfish helper의 양성 인수를 구분한다.
CPU04가 성공하더라도 CPU03의 외부 helper 실패 여섯 건을 대체하지 않는다.
조건부 C slot·Repair 수순 비교의 합성 정확성, 실제 모델 소비, 전략적 품질,
전체 엔진 대국도 각각 별도 인수다. PR은 Draft이며 전체 목표는 아직 미완료다.

## 재개 시 범위 대조와 CPU04 준비 감독기 소스

사용자는 이번 GPU 검증을 보류했다. 실제 학습 제외 조건도 유지한다. GPU 실행을
하지 않았으며, 메모리 여유와 actual81 소유권 회수가 확인되기 전에는 WSL 모델 실행·
CPU 신경망 대국을 재개하지 않는다. 경량 Windows stdlib 메타데이터 검사와 소스 작업은
별도 범위다. 원래 실패 자료·lease·보존 산출물을 성공 상태로 고쳐 다음 실행을 열지 않는다.

CPU04 준비의 자기 보고와 실제 종료를 구별하는 총괄 감독기 초안을 보존 루트에 작성했다.
freeze revision 3의 정확한 원문 bytes를 로드하고, nonce-bound oneshot 서비스에서 준비를
실행하도록 연결했다. 독립 main PID·start ticks·실행 이미지·Invocation·cgroup·자원 관측,
actual wait client exit·stdout/stderr EOF·소유 cgroup 부재와 source 전후 일치가 모두 필요하다.
로그 저장 실패는 서비스 정리를 막지 않으며 양성 admission은 거부한다. 원래 300초 작업과
30초 정리의 전체 창은 갱신하지 않는다. 최종 출력 후에도 deadline·취소를 다시 확인한다.
시간 필드는 게시 전 범위이며, 호출한 총괄의 실제 supervisor exit·EOF·전체 시간·게시 pin
인수가 끝나야 대국 runner를 시작할 수 있다. 이 실제 Linux 연결 인수는 아직 미실행이다.

수정된 감독기 소스 `a08e10407d80b59a07393bc59ec9cdea9a0301f961207358947c6c08166939ad`
21,685 bytes를 대상으로 Windows Python 3.10 stdlib의 syntax/module 및 순수 finality
메타데이터 37개 검사를 실제 실행해 모두 통과했다. 기존 CPU04 revision 3의 raw loader·
publication 31개 검사와 별도 결과다. 이 검사는 실제 서비스·O_NOFOLLOW·권한 전환·
모델·GPU·물리 정리·admission 발행을 관측하지 않았으며 준비나 대국 인수로 승격하지 않는다.

전체 원문의 구현 목록을 다시 대조하여, 최소 CPU pilot과 전체 계획 완료를 구분한다.

| 항목 | 현재 범위와 남은 작업 |
|---|---|
| CPU04 최종 인수 | 준비 감독기와 typed Core·실신경망·시계·PGN·수명·저장 인수는 미실행. 실패한 `go`의 알려진 정수 계수 0을 요구하며 CPU03 자료로 대체하지 않는다. |
| CPU 캡처 정렬 | 기존 MVV/LVA는 원문 기본 SEE와 다르다. legacy identity를 보존한 명시적 `LegalSeeV1` 구현과 consumer 연결·인수를 별도로 진행한다. 제한 교환 점수는 전게임 bound가 아니다. |
| 학습 준비용 V 환류 | 기존 부모당 한 번의 cold V→CPU→label은 새 CPU 결과→새 immutable 과제 입력→V 판단 갱신→다음 작업의 반복을 대신하지 않는다. 별도 bounded coverage feedback 경로를 준비한다. 기존 seal·reader와 제품 V-free 경로는 보존한다. |
| CUDA record별 공개 K/V bank | host record pages와 CUDA whole-input cache는 존재하지만 resident record pool과 실제 device join은 아직 구현되지 않았다. GPU 검증 보류와 구현 누락은 별개다. |
| CUDA private Warm | 현재 CPU P/C-only이며 CUDA는 명시적 미지원이다. 원문의 별도 근사 모델 실험으로 추적하며 Fresh CPU04의 선행 조건으로 추가하지 않는다. |
| Repair 후 C 재검토 | 수선·CPU 검증·의존 결론 갱신은 존재한다. 수선한 수순을 대상으로 한 C 재공격과 원문 재검토 트리거의 연결 여부를 추가 확인한다. 확인 전에는 구조적 누락이나 전략 효과를 단정하지 않는다. |

set-associative TT·paused node-stack resume·host KV tier·압축·overlap·세 CPU 프로필 전체는
원문의 제안·선택·미결정과 구분한다. 현재 completed-iteration resume를 paused stack
resume라고 부르지 않는다. 실제 학습을 제외한 전체 목표는 여전히 미완료다.

이번 재개에서 원격 PR 23 HEAD `ea7d567cdb26a2ae387eacc29224612a4fcb0b75`의
Linux·Windows·CPU bindings·PALS model 네 CI 성공을 다시 확인했다. 이 성공은 해당
소스의 CPU 검사이며 위 감독기와 이후 SEE·V 환류 소스의 성공 결과가 아니다.

### 명시적 LegalSeeV1 소스 연결

`CpuOrderingPolicy::LegalSeeV1`과 별도 constructor를 작성했다. 기존 constructor·
세 CPU profile·CpuConfig·CPU value namespace·legacy search conditions는 유지한다.
새 정책은 `rz-cpu-pvs-legal-see/0.1`과 ordering·물질값·32 ply·정렬당 4,096개
교환 상태·원래 deadline/cancel·실제 node 비용을 별도 conditions로 기록한다.
resume는 전체 conditions가 다르면 거부하며 TT는 정책이 불변인 해당 engine이 소유한다.

SEE는 기존 Rules의 합법 recapture·네 승격·앙파상·핀·왕 안전성과 실제 move delta를
사용한 같은 target의 제한 물질 교환 minimax다. abstract decline의 0은 교환 게임의
선택일 뿐 미관측 CPU 값을 0으로 채우는 동작이 아니다. mate·전게임 bound·pruning
권한으로 사용하지 않는다. 각 entered 교환 상태를 기존 전체 node 한도에 청구하고,
취소·마감·상한 실패는 typed 미완료/오류로 남긴다. score 계산은 한 수당 한 번이며
모든 score가 확인된 뒤에만 move 배열을 바꾼다. 중단된 score를 MVV/LVA로 대체하지 않는다.

교환 정확성 10개와 namespace·resume·정렬·수명·상한·부분 비용 연결 6개의 Rust 검사
소스를 추가했다. 총괄의 직접 검토와 diff check를 수행했다. 루트의 edition 2021과
`rz-search`의 명시적 edition 2024를 혼동해 최종 2021 formatter를 적용한 결과,
`55fcc7657dbcb397a2e0f9bdcd9e9f0a4f967dfb` CI의 bindings·Linux·Windows는
서식 검사에서 실패했다. 원래 실패 로그를 보존했고, 실제 crate edition 2024로
다시 정렬한 해당 source rustfmt check는 통과했다. 새 SHA의 CI 인수는 별도다.
메모리 조건 때문에 로컬 Cargo·모델·WSL 검사는 실행하지 않았으며 새 source의 CPU CI를
별도로 인수한다. CPU_T CLI/producer의 실제 선택과 비교 명세 연결은 아직 미완료다.
이 opt-in을 기존 CPU04의 legacy binary/config에 소급 적용하거나 기력 개선으로 보고하지 않는다.

문서 커밋 `1c9ef4024bc73623dfb501eca51ba656872b0cf8`의 네 CPU CI도 실제 SUCCESS를
확인했다(Windows 최종 2026-10-08 02:10:59 UTC). 이는 이후 LegalSeeV1 소스의 검사 결과가
아니다. 준비 감독기는 추가 cleanup 관측 보완 후 22,121 bytes,
`dd49cbf64606265c3fef2e4d4b4f45a542eb531a0ea24e668f23fe9454dd43f6`로 원문을 별도
보존했고, 같은 순수 메타데이터 37개가 모두 PASS였다. 물리 완료·대국 검사는 미실행이다.

### SEE source 후속 CPU CI 관측

`6c166dc54a68cbdc739aaebf1147d2318e811b53`의 기존 workflow
run `37717186953`은 Linux·Windows·CPU bindings·PALS model CPU 네 job 모두
SUCCESS로 종료됐다. Windows의 최종 완료는 2026-10-08 02:24:47 UTC다.
이는 실제 crate edition 2024로 수정한 SEE source와 CPU 검사 결과다.
이전 `55fcc76`의 서식 실패는 보존한다. 로컬 WSL·모델·GPU·대국을 새로
실행한 결과가 아니며, CPU_T CLI/producer의 명시적 SEE 선택 연결과
실제 CPU04 pair·전체 V 환류·CUDA record bank·Repair 후 C 재검토는 별도 잔여다.

### Private V coverage feedback 첫 소스 단위

`verifier_feedback.py`는 별도
`rz-pals-private-v-feedback-plan/1` 도메인의 고정 stage를 받는다. 이전 실제 CPU
request·response·process·raw ledger를 보존하고, 같은 parent·Rules·순서·branch·
binary·profile·H/N/Q/TT/wall 조건에서 완료된 iteration 깊이만 다음 새 checked
semantic header에 넣는다. 새 input/before/query seal 이후 기존 frozen V를 cold
forward해 다음 작업을 선택한다. raw hash를 모델 특징으로 넣지 않는다.

Global/per-parent stage·node·wall·output·semantic owner bytes·FLOPs를 발주 전에
예약한다. 원래 task wall을 확보하지 못하면 unresolved로 종료하며 예산을 조용히
낮춰 같은 coverage로 재해석하지 않는다. 초기 stage는 legacy
`resume_task`/`cross_profile_recheck`/`defer`에 한정하고 SEE를 자동 소비하지 않는다.
재로드는 원시 ledger와 새 입력의 연결·등록 pin·imported source·관측 범위를 다시
대조하는 `CheckedFeedbackHistory`를 반환한다. ordinary dataset·positive training
capability가 아니며 실제 producer 종료와 전체 시간·cleanup의 외부 인수가 필요하다.

동결 첫 source는 72,395 bytes,
`311eb5a61e8488194d53b4093d2338de4d3d39348dd7cb37f895d3d5dee026c0`이고
20개 집중 fixture source는 23,702 bytes,
`3413ed5d05509a27df6d166c855f75963831d6efe95e8f1f1c564187be10994d`다.
root100의 CPython 3.10 AST 검사는 실제 exit 0과 source 전후 동일을 확인했다.
module import·fixture·Torch·Rust child·GPU 실행은 하지 않았다. CPU CI와 독립
source review를 후속 인수한다. fixture의 synthetic child/Rules 자료 아래 두 cold
V forward 검사는 실제 등록 Rust 실행이나 Rules 재생 증거로 확대하지 않는다.

이 단위는 declared-snapshot-relative completed-iteration coverage feedback이며
전체 전략적 V 우선순위·budget head·private Warm·CPU stack 복원·학습 완료가 아니다.
task rank·WDL·ordinary/positive target은 계속 masked/false다. P/C 제품은 V-free다.

### CPU_T private SEE 소비자 연결

Private `pals_cpu_task` example은 명시적 `--cpu-ordering=legal-see-v1`과
봉인된 `ordering_policy: legal_see_v1` 요청을 함께 요구한다. task domain은
`rz-pals-private-cpu-task-legal-see/1`, conditions domain은
`rz-pals-private-cpu-conditions-legal-see/1`이며 search identity는
`rz-cpu-pvs-legal-see/0.1`이다. marker 생략·무인수·기존 capabilities는 legacy를
유지하고 null·unknown·다른 namespace·다른 mode의 혼합은 검색 전에 거절한다.

Producer의 `ordering_policy` 선택과 explicit bank reload를 함께 연결했다.
기본 helper·value·후보/semantic/continuation·native PALS·CPU_R·V coverage loop는
legacy를 유지한다. report와 completed-iteration resume는 실제 선택한 engine의
전체 조건을 대조한다. raw score를 새로운 reward나 WDL로 바꾸지 않는다.

새 집중 fixture source는 Rust bridge 7개·CLI parser 1개·Python 7개다.
root는 두 Rust 파일을 해당 crate edition 2024로 서식 정리했다.
로컬 Cargo·Python import·fixture·모델·WSL·GPU 실행은 하지 않았다.
이 consumer와 새 V feedback의 정확한 integration SHA를 기존 CPU CI에서 검사한다.

### Private SEE 소비자의 첫 CI 실패와 기본 호출 호환성 수정

`7fbb389`의 CPU CI run `37719631256`은 Linux workspace, Windows workspace,
CPU bindings 세 job에서 성공했지만 `pals-model-cpu-correctness`는 실패했다.
PALS Python suite는 339개 검사를 실행했고 기존 `test_current_consumers`의
다섯 사례에서 `ordering_policy` 키워드를 받지 않는 legacy bridge fixture와
producer 호출이 충돌했다. 이 실패를 모델 수치 성공이나 전체 CI 성공으로
집계하지 않는다. 원래 job log와 exact-SHA 결과는 저장소 밖 인수 자료로 보존했다.

후속 수정은 기본 legacy producer의 기존 bridge 호출 형태를 유지하고,
명시적으로 선택한 SEE lane에만 새 ordering 키워드를 전달한다. 기존 consumer
검사를 완화하거나 fixture에 임의 키워드를 받아들이게 하지 않는다. request와
report의 ordering domain, 실제 CPU conditions 및 raw evidence 검증은 그대로다.
이 수정의 소스·후속 CI 결과는 `7fbb389`의 실패 기록과 별도로 인수한다.
로컬에서는 Python import·모델 실행·Cargo 검사·GPU 검사를 시작하지 않았다.

후속 source `2766a42e6fd9153cdb9d5ccb4175ed29ad64f70f`의 CI run
`37720496417`은 네 CPU job 모두 SUCCESS를 확인했다. Windows의 마지막 완료는
2026-10-08 03:05:25 UTC다. PALS Python suite 339개와 CPU forward/export/no-step
준비는 해당 source의 검사이며, 독립 검토에서 드러난 feedback reload 누락의
후속 수정이나 새 packing artifact 검사를 이 성공으로 소급 인증하지 않는다.

### Record device packing artifact의 소스와 CPU 수치 검사 준비

별도 `device_packing_artifacts.py`는 학습 가중치가 없는 deterministic ONNX
IR10/opset17 결합 그래프와 strict manifest를 제공한다. 기존 learned public/role
export와 CLI·native manifest를 바꾸지 않는다. B1 FP32, K/V 2 heads × 64,
board 66 tokens, record 최대 128개를 명시하며 입력 387개·노드 275개·INT64
initializer 5개/40B를 사용한다. ordered record occurrence·중복·source offset을
보존하고 0-record에서는 실제 zero-feature projection token과 false mask를 요구한다.
K/V를 임의의 0으로 합성하지 않는다.

CPU 입력 witness는 전체 readonly owning backing·feed alias·각 고유 payload와
제어 배열·mask를 함께 검사한다. 노드 dtype/shape의 최대 payload 합계를 산술로
계산하되 allocator·session·workspace 또는 peak 관측값으로 보고하지 않는다.
출력은 저장소 밖 새 절대 경로에만 생성하며 기존 경로를 덮어쓰지 않는다.

초도 독립 검토에서 재검사의 model/graph/projection identity가 caller 객체의
custom equality에 의존할 수 있는 타입 누락을 발견했다. 두 block identity와
모든 expected projection key를 exact lowercase SHA-256 문자열로 검사하도록
보강했으며 prepare·직접 구성·replace 경로의 거절 fixture를 추가했다. 후속 독립
읽기 전용 검토는 해당 우회가 소스상 닫혔음을 확인했다. 동결 source는 25,949B
SHA `c95c9768f8072db7da3165169bc37923f51cde9bc8b2e0b59ca70564864d54f5`,
test source는 17,038B SHA
`36e2defb197c973f90a88ac3530fc6f37e4e70165c413c2085b674b1eacc9faa`다.

13개 검사 method 소스는 exact graph/manifest, 합성 K/V의 bitwise routing,
0/mixed/128 record·중복·signed zero·visible finite, 타입·owner·예산 거절을
포함한다. root104는 동결 pin 전후 일치와 Windows stdlib AST를 실제 exit 0으로
확인했으며 import·검사 실행·graph 생성·모델·WSL·GPU 실행은 하지 않았다.
실제 CPU 수치는 후속 exact-SHA CI에서 인수한다. learned public projection의
수치 대조, Rust native consumer, CUDA resident owner·lease·fence·quarantine은
이 단위에서 미구현/미인수다. GPU 검증은 계속 사용자 보류다.

### V coverage feedback reload의 세 검토 누락 수정

초도 `7fbb389` 이후 독립 원문 검토에서 발견한 세 누락을 후속 소스로 수정했다.
loader와 registered asset의 첫 I/O 전·각 read 전후·replay 전후에 기존 유한
deadline 검사를 재사용하여 bool·NaN·무한대·만료를 거절한다. live producer와
reload는 같은 round cost와 원자적 예약 함수를 사용한다. 각 round에서 `2N`,
FLOPs·semantic bytes·`6 × childcap + 2MiB` 출력 비용을 먼저 예약하고, 검증된
실제 nodes와 저장 bytes의 차이만 환급한 뒤 다음 round를 인수한다. 작은 최종
실제 합계로 발주 전 불가능했던 예약을 숨기지 못한다. 전체 stage의 H/N/wall과
child output cap도 같은 actual-capability gate로 검사하고 Defer 이후 round를
거절한다.

후속 읽기 전용 독립 검토는 세 제품 누락이 소스상 닫혔음을 확인했고 추가
must-fix는 발견하지 않았다. 동결 feedback source는 74,836B SHA
`c06f0d31ef61874f95f7205f4fa0ce29427a65b189ff821d8c25f7088d63dd8c`,
test source는 33,316B SHA
`d9a9b3b7692d62327550d94b2e632b2f349312be02cb0ee65730e4ab094c177d`다.
source9는 실제 dependency bytes를 전후 검사하므로 SEE consumer의 후속 수정도
실제 producer source pin으로 대조하며 상수 pin을 자동 재해석하지 않는다.

검사 소스는 기존 20개와 추가 7개로 총 27개다. 노드·출력 예약 음성 사례는
plan·before·dispatch·decision·receipt bytes를 다시 봉인하여 해당 예약 gate를
검사한다. Defer 사례는 decision과 pin을 재봉인해 episode 종료 gate 도달을
검사하지만 나머지 CPU request/evidence는 기존 Resume 상태다. 따라서 이것을
전체 필드가 일관된 Defer→후속 round 역사의 end-to-end 검사로 표현하지 않는다.
root105는 동결 pin 전후 일치와 Windows stdlib AST를 실제 exit 0으로 확인했다.
제품 module import·fixture·Torch·실제 Rust child·WSL·GPU 실행은 하지 않았다.
수정 전 source/CI는 보존하며 실제 27개 검사·전체 producer 종료와 cleanup은
별도 exact-SHA 인수 대상으로 남긴다. ordinary/positive target·실제 학습·전체
전략적 V 환류로 scope를 확대하지 않는다.

packing source `9fbcdcd145c26e488fad44641ad794174859fa00`의 CI run
`37721184850`은 네 CPU job 모두 SUCCESS를 확인했다. Windows 최종 완료는
2026-10-08 03:13:43 UTC다. PALS suite 352개가 통과했고 packing의 13개 method도
skip 없이 실제 `ok`를 확인했다. CPU ORT의 합성 K/V 수치·artifact·입력 거절
검사는 이 source에 한정하며, 위 feedback reload 수정은 후속 SHA에서 검사한다.
학습·GPU·Rust CUDA resident consumer·CPU04 pair 인수로 확대하지 않는다.

feedback reload 수정 source `cb0752fc95ad70f3ad76d9fd6c9cf308429c0f61`의
CI run `37721828446`도 네 CPU job 모두 SUCCESS를 확인했다. 마지막 Linux job은
2026-10-08 03:19:48 UTC에 완료됐다. PALS suite 359개와 feedback 27개 모두
실제 `ok`를 확인했으며 skip으로 대체하지 않았다. 두 cold V forward·raw replay
fixture는 합성 child 관측과 frozen CPU 모델을 사용한다. 실제 등록 Rust child,
외부 owner의 producer 종료·cleanup 및 CPU04 대국의 인수는 계속 별도다.

### Repair 뒤 같은 line을 한 번 재검토하는 opt-in 경계

`PostRepairRecheckPolicy::SameRepairedLineOnceV1`은 실제 Repair 출력이 수용되고
완료 evidence를 가진 해당 repaired line에만 적용한다. 기본 constructor는
`Disabled`이며 기존 search identity를 보존한다. 활성 정책은 별도 search identity와
immutable getter를 가지며 첫 지원 범위는 자체 CPU/Rules evidence다. ExternalUci
checker와 이 정책의 조합은 생성 단계에서 거절한다.

현재 root·history·line·record revision·checker namespace를 대조하고, 이번 Repair를
유발한 refutation과 다른 우리 응수 뒤의 상대 anchor에서 Reply를 최대 한 번 요청한다.
이후 suffix는 Rules로 재생하며 추가 Reply follow나 재귀 Repair를 만들지 않는다.
같은 길이의 완전 수순에서 양쪽 Rules terminal 또는 같은 namespace·같은 완료 CPU
depth를 만족할 때만 해당 repaired LineId의 조건부 refutation을 게시한다. 미완료·단축·
불법 suffix·깊이 불일치·대안 부재는 새로운 반박이나 전체 방어 증명이 아니다.

최종 publication은 observation 추가·line refutation·지원 counter 증가 전에 다시
deadline/cancel을 확인한다. 독립 검토에서 발견한 이 guard 누락은 수정했으며,
focused fixture는 바로 이 최종 경계에 늦은 취소와 만료를 넣는 형태다. 실제 CPU·NN
실행 인수로 해석하지 않는다. CLI/driver·manifest/arena·collector의 선택과 receipt
연결은 별도 소비자 단위이며, 이 engine 단위만으로 제품에서 선택 가능하다고
보고하지 않는다. 특히 early/unknown PalsResult에는 정책 식별 필드가 없다.

이 engine source `2367174552e3553d9bc11168e410b2b5cf96bf74`의
CI `37723109129`에서는 Linux·Windows Rust 및 CPU bindings 성공과 Linux의
post-repair 집중 검사 7개 실제 `ok`를 확인했다. Windows는 2026-10-08 03:38:28 UTC에
완료됐다. PALS Python suite는 359개 중 native slot Repair 소비자에서
16개 실패·1개 오류로 끝났다. 기존 닫힌 source-pair 등록이 이전 engine byte pin만
허용하여 새 source를 `unreviewed_collector_transition_source_pair`로 거절한 것이다.
이를 일반 NN/대국 성공이나 새 정책 소비자 연결 실패와 혼동하지 않는다. 새 engine의
기본 Disabled 경로와 변경되지 않은 collector의 실제 constructor를 대조한 후에만
새 exact source pair를 등록하며, 미검토 source 거절과 과거 pair 읽기를 유지한다.

### Device record planner와 전체 backing 소유 예약의 첫 Rust 단위

`rz-eval`의 `device_pages_plan`은 sealed backing·certified slice·순서 있는 fixed
ports·전체 owner bank·유한 예약을 분리한다. 현재 유일한 구현은 immutable CPU
fixture backing이다. CUDA namespace는 선언이며 CUDA owner factory, packing Run,
private P/C native consumer, device fence·물리 완료·quarantine 연결은 아직 없다.
기존 두 graph manifest guard와 CLI·native 통계를 변경하지 않는다.

projection reference는 비소유 key·generation·offset이며 실제 plan만 고유 whole
owner pin을 획득한다. 중복 record의 입력 순서와 port는 유지하고, 전체 backing
capacity를 한 번 청구한다. 인증한 slice만 게시하며 0-record view는 실제 zero-feature
projection과 false mask를 요구한다. cache hit에서도 pending public block·join·내부
node·control·private output overlap·세 session의 예약은 줄이지 않는다. session 및
추가 metadata의 unknown/0 선언은 성공적인 0-byte 값으로 대체하지 않는다.

독립 검토에서 같은 namespace의 다른 registry가 외부 plan을 인수하면 그 plan이
보유한 backing 비용을 누락하는 문제를 발견했다. private checked instance seal을
registry·plan·Copy offset에 함께 묶고, actual origin을 lookup·pin·예약 전에 검사해
수정했다. 의미 projection/cache namespace에는 seal을 넣지 않는다. UID overflow는
typed 거절이며 clear 또는 이후 할당 실패에서 ID를 재사용하지 않는다.

focused Rust fixture는 기존 10개와 cross-registry·재구성 backing·UID 소진 3개를
합쳐 13개다. 수정본의 독립 delta 검토와 파일별 Rust 2021 fmt는 통과했으며 실제
fixture 실행은 후속 CI에서 인수한다. packing node 최대 payload 산술 합 1,142,291 B,
그중 별도 joined/final owner를 뺀 내부 합 943,634 B는 정적 예약 근거다. allocator,
workspace, 실제 peak 또는 VRAM 측정값으로 해석하지 않는다.

기존 CPU04 frozen preparation과 historical binary는 이번 engine·UCI·backend source
변경 후 현재 source 등록으로 재사용하지 않는다. 실제 준비를 재개할 때 source 전후
pin과 새 binary 등록을 다시 확인하며, 이전 명세와 실패·중단 자료는 보존한다.

### 재검토 정책의 UCI 선택과 실행 영수증

UCI는 `--pals-post-repair-recheck=same-repaired-line-once-v1`의 명시적 선택만
새 정책 constructor로 전달한다. 생략 또는 API의 Disabled는 기존 v1/v2 session
hash 경로와 receipt 생략 형식을 보존한다. 중복·unknown 값과 external checker
조합은 profile·모델 로딩 전에 거절한다. driver는 실제 engine의 immutable 정책·
search identity·조건 getter를 캡처하고 role 실행 입장에서도 등록값과 대조한다.

새 session domain은 전체 checker namespace와 선택 identity를 함께 결합한다.
optional `pals_search_policy`는 version·policy·search_identity·conditions_sha256을
담고, legacy `None`은 직렬화하지 않는다. marker는 zero-work·early·실패에서도
startup 선택만 나타낸다. 실제 재검토 실행·수선 성공·조건부 refutation 수를 만들거나
기존 resolver·모델·완료 counter를 그 증거로 바꾸지 않는다.

main 3개·driver 5개·work 3개 총 11개 focused fixture 소스를 준비했다. 독립 소스
검토·파일별 서식과 후속 CI를 구분한다. arena/manifest의 새 lane 인수와 collector의
독립 constructor·producer 등록은 아직 연결되지 않았다. 일반 V3 paired 인수나
학습 자료 소비까지 연결했다고 보고하지 않는다.

통합 source `ebde32c5b4dbb1ef9528a47f0b106753b1f41edd`의 CI
`37724974142`는 CPU bindings만 SUCCESS였고 Linux·Windows Rust와 PALS model은
FAILURE였다. Rust는 UCI focused fixture의 `OwnedCpuChecker` 반환 타입에 실제
`CpuEngine` type argument가 빠져 E0107로 중단됐다. 해당 fixture 서명만 수정하며,
13개 registry 또는 11개 UCI 검사를 이 실패 run에서 실행한 것처럼 표시하지 않는다.
PALS model은 별도의 이전 closed source-pair 거절 16개 실패·1개 오류였다. 원래 run의
실패·로그·종료 결과를 보존하고, 후속 통합 SHA에서 다시 인수한다.

### 기본 collector의 닫힌 native Repair source-pair 등록

기존 engine `4469f9d…`·native `564d2b29…` profile은 유지하고, 실제 읽은 새 engine
`9c0de959…`와 변경되지 않은 native의 exact pair를 별도 profile로 추가한다. native의
실제 constructor는 여전히 Disabled를 선택하며 새 recheck의 첫 분기에서 즉시
반환한다. 실제 build manifest·source bytes·commit·등록 binary 결합을 먼저 검사한
뒤 closed pair를 선택하고 returned pin은 detached copy로 반환한다.

독립 검토에서 실제 `pals_search_policy` 키 누락과 nested `search_version` 검사 누락을
발견해 수정했다. source·native·search_configuration·independent_registry의 같은
typed owner 목록으로 marker와 version을 대조한다. 현재 collector가 만들지 않는
marker는 Disabled 선언이어도 거절한다. present 새/unknown/null version은 거절하며
기존 version과 생략은 유지한다. 자체 CPU namespace는 별도 cpu_task_source이므로
이 검사를 그 CPU version으로 대체하지 않는다. 새 opt-in collector는 미지원이다.

focused Python fixture는 36개이며 실제 marker 16개 및 nested version 12개 음성
subcase를 포함한다. historical pair selector의 합성 seam을 실제 과거 producer·빌드
재검증으로 해석하지 않는다. root109는 최종 byte pin과 stdlib AST만 PASS다. 실제
import·fixture·consumer 인수는 아래 후속 CPU CI에서 확인하며 local NN·WSL·GPU는 미실행이다.

### `9fa1505`의 통합 CPU CI 인수

[CI 37725468882](https://github.com/daejunnom/RoveZero/actions/runs/37725468882)는
source `9fa1505481bc3acf7e1a1c4b3af4b8c036048b25`에서 Linux·Windows workspace,
CPU bindings, PALS model CPU 네 job 모두 SUCCESS다. 최종 Windows job은
2026-10-08 04:07:55 UTC에 완료됐다. Linux·Windows의 all-targets/all-features 검사,
native CLI, Rules release 대조, Python 도구 검사와 clippy `-D warnings`를 포함한다.
Linux 로그에서 record planner/registry의 집중 검사 13개와 post-repair engine 검사
7개, 새 UCI 선택·journal 검사의 actual `ok`를 확인했다. PALS model suite 365개와
native Repair 소비자 집중 검사 36개도 실제 실행돼 통과했다. 이전 `2367174`와
`ebde32c`의 실패 기록을 소급 성공으로 바꾸지 않는다.

이 인수는 현재 구현 소스의 CPU 검사다. CUDA backing·물리 device join·활성 collector,
전체 전략적 V, 실제 GPU 수치·메모리·대국을 완료했다고 표시하지 않는다. 후속
arena/manifest 연결은 별도 source 단위로 검증한다. 기존 CPU04 preparation/binary의
source 등록은 현재 HEAD로 재사용하지 않으며 GPU 보류·로컬 heavy CPU 자원 질문의
답변 대기·실제 학습 제외·전체 목표 미완료·Draft 경계를 유지한다.

### Repair 재검토 정책의 V3 명세와 arena 소비 연결

명시적 `pals-post-repair-recheck/1` 검색 명세는 단일 옵션
`post_repair_recheck=same-repaired-line-once-v1`과 자체 CPU_R만 허용한다. 실행 recipe의
동일 선택을 대조한 뒤 UCI flag 하나를 마지막에 추가한다. 미선택 recipe의 argv 순서와
생략된 optional 필드, V3 canonical bytes 및 과거 V1/V2 읽기 경로를 보존한다.
새 optional의 present-null과 미지원 선택은 거절하며, 기존 optional의 null 의미는
바꾸지 않는다. `rz-experiments`는 engine/UCI에 역의존하지 않는 얇은 wire 타입을
소유한다. arena의 집중 검사 소스는 현재 engine의 조건 literal SHA와 계약 pin을 대조한다.

work 기록 감사와 공개 native 단독 감사는 startup·termination의 실제
`search_work.pals_search_policy` 네 필드를 모두 대조한다. 새 lane에서는 두 marker가
필수이고 legacy·자체 CPU·외부 UCI·외부 CPU_R 경로에 주입된 marker는 거절한다.
Core에는 명세의 기대값을 복사하지 않고 실제 work 관측에서 검증한 marker를 옮긴다.
marker가 맞더라도 실패한 go·불명 작업·물리 완료·buffer 해제·NN 소비·시계·PGN의
기존 인수 조건은 유지한다. 선택을 Reply 실행·재검토 성공·전체 방어 증명으로 집계하지 않는다.

검사 소스는 manifest 신규 10개와 arena 신규 6개, 기존 external 선언 fixture 보강
1개다. 최대 옵션의 CUDA recipe는 metadata 대조 대상으로 legacy 22개 argv에서
선택 후 23개가 되며 기존 32개 상한을 유지한다. 실제 CUDA 실행 자료는 아니다.
원문 동결·독립 검토·서식·CPU CI의 증거를 각각 구분하고, 이번 단위 이전 `9fa1505`의
성공을 새 arena source의 실제 실행으로 재사용하지 않는다.

native collector의 활성 정책 constructor·독립 출처 등록과 실제 Repair→Reply 인과
trace는 후속 단위다. 일반 Reply의 물리 완료·delivery·consumption만으로 특정
Repair 재검토를 증명하지 않는다. 같은 repaired line·revision·상대 anchor·CPU
namespace 및 실제 비교 evidence를 결합해야 하며, 기존 Disabled collector와 label
인수 범위를 완화하지 않는다. 전체 전략적 V와 CUDA record bank도 계속 미완료다.

### 재검사 observer와 native journal 연결 상태

`5933f65`의 engine은 prepared/finished 기본 noop과 Box 전달을 제공한다. 실제
accepted Repair record와 수순·Rules snapshot, 저장된 CPU Observation/Execution ID와
TaskRecord, 다음 Reply 결과 및 조건부 publication을 빌려 전달한다. provenance 없음은
`Unobserved`, 실제 handle 모순은 `InvalidProvenance`이며 수동 값을 완료 작업으로
만들지 않는다. Reply 호출 시도는 물리 NN 제출이나 완료와 별도다.

종료 callback 뒤 두 경로가 성공했을 때 원래 deadline/cancel을 다시 확인한다.
원래 오류와 첫 observer 보조 오류의 우선순위는 유지하며 이미 적법하게 게시한
조건부 기록은 철회하지 않는다. [CI 37732758483](https://github.com/daejunnom/RoveZero/actions/runs/37732758483)의
Linux·Windows·CPU bindings는 성공했고 observer 집중 검사 11개가 두 OS에서 각각
실제 통과했다. model CPU job은 닫힌 source-pair 미등록과 이전 source literal에서
실패했으므로 이 commit을 전체 CI 성공으로 기록하지 않는다.

후속 native journal은 마지막 Repair 한 호출 대신 repaired suffix의 각 실제 accepted
요청·입력·sidecar·lineage·원시 policy 순위를 결합한다. 별도 정책 원문 등록을 실제
engine 선택과 대조하고 `prepared → reply_bound → finished`의 출력 공간을 Reply 전에
예약한다. 기존 lineage domain은 유지하며 새 descriptor 연결은
`native-recheck-traces.jsonl`에 보존한다. 최종 consumer는 원래 receipt inventory와
실제 producer journal 및 두 Rules 재생 endpoint를 함께 확인해야 한다. marker나
일반 Reply 완료만으로 실제 재검사·Repair 성공·WDL·학습 target을 인정하지 않는다.

CUDA packing의 첫 연결부는 `RegisteredPackingArtifactBytes`의 실제 바이트 pin과
고정 선언 검사다. 기존 Python 고정 wire와 독립 대조했으며 Native graph body,
weight-free 내용, session/provider·수치·fence는 `NotPerformed`다. 이는 resident
record bank·native join·P/C read 구현이나 GPU 성공의 대체물이 아니다. 두 후속 단위의
실행 검사는 최종 통합 commit의 CPU CI에서 별도 인수한다. 이번 GPU 보류, 로컬
heavy CPU 자원 답변 대기, 실제 학습 제외 및 Draft 경계는 유지한다.

### 원래 receipt에 결합한 native recheck 전체 관측 검증

`native_recheck_witness.py`는 닫힌 opt-in engine/native source pair를 독립 literal로
등록한다. 원래 build·source·binary 등록과 최초 receipt 전체 inventory를 대조하고,
Repair suffix의 모든 실제 요청, 다음 Reply binding·raw policy·물리 완료·소비,
두 전체 Rules 수순, 저장된 OwnCpu TaskRecord/Observation과 조건부 Model/Estimate
publication을 공개 factory에서 함께 검사한다. 기존 receipt에 trace를 사후 추가하지
않는다. marker·소스 hash·Reply 완료 하나만으로 재검사 관측을 인수하지 않는다.

반환 범위는 `conditional_same_repaired_line_observation_only`다. WDL·Rules 증명,
전략 repair·counterexample·policy/D ranking·학습·제품 권한과 최종 search envelope
완료 권한은 모두 false다. 취소·마감·원래 서비스 오류·미완료·Rules capability 부재는
unresolved로 보존하고, source·raw·요청·관점·게시 모순은 거부한다. observer 이후의
최종 search 상태를 관측했다고 주장하지 않는다.

59개 source test methods 중 공개 whole 경로 11개는 별도 최초 receipt fixture와
실제 strict loader·공개 anchor/whole factory를 사용한다. build·process·Rules·OwnCpu
실행 기록은 합성이며, 실제 읽은 Rust source bytes와 구분한다. 공개 양의 whole
사례는 짝수 ply이고 홀수 관점·duplicate/sign 검사는 기존 helper 범위다. 두 독립
정적 검토와 AST 검사는 실행 성공을 대신하지 않는다. 실제 통과는 후속 CI에서 인수한다.

직전 [CI 37736002427](https://github.com/daejunnom/RoveZero/actions/runs/37736002427)는
`31cd1ca44bc2b735f9923047f586069162921adf`의 Linux·Windows·CPU bindings가 성공했다.
model CPU job의 366개 검사 중 한 역사 profile fixture가 현재 source pin을 이전
observer pin과 혼동해 실패했다. 후속 수정은 이전 literal pair 검사를 별도로 보존하고
현재 pair를 정확히 지정한다. 이 수정이나 새 whole 검사를 이전 CI 성공으로 소급하지 않는다.

후속 [CI 37738287013](https://github.com/daejunnom/RoveZero/actions/runs/37738287013)의
`9c1d57cd13b04b2e993c046c58863b5af102e9ef`는 Linux·Windows·CPU bindings가
성공했고 공개 whole 11개도 실제 통과했다. 전체 model suite 425개 중 선택적 빈
JSONL 파일을 기존 required-row parser로 읽은 세 경계가 error여서 run은 실패다.
후속 수정은 trace·public source·raw event·raw output의 exact empty bytes만 빈
관측으로 처리한다. required prepared/producer 파일과 nonempty JSONL의 완전 줄바꿈
검사는 유지하며, empty가 완료·소비·target 권한을 만들지 않는다. 별도 nonempty
malformed 회귀를 더해 source methods는 60개다. 후속 CI 성공은 아직 미인수다.

### Rust의 고정 packing graphbody 검사

`CheckedFixedPackingGraph`는 admitted immutable pair를 이동 소유하고 실제 ONNX
body를 닫힌 recipe와 대조한다. 다섯 INT64 controls, 387 inputs·3 outputs·272
value_info·275 ordered nodes의 dtype·shape·이름·edge·attribute와 finite gate 연결을
검사한다. wire field/packed element·depth·fallible allocation의 유한 검사를 적용하고
retained backing과 inspection scratch 예약을 나누며 관측 peak로 보고하지 않는다.

두 독립 정적 검토와 file-specific rustfmt를 마쳤다. 12개 synthetic wire fixture는
repinned foreign body·연결·finite gate·control·type·shape·unknown/stateful field·중복
및 malformed/resource 경계를 다룬다. 실제 Python artifact interop은 별도 인수다.
Rust는 의미가 같은 packed/unpacked/zero 생략 wire를 허용하므로 Python deterministic
export의 byte equality와 구분한다. `native_verification=NotPerformed`는 유지하며
session interface·provider·실제 finite 출력·Run/fence·resident bank를 인수하지 않는다.

후속 [CI 37739736920](https://github.com/daejunnom/RoveZero/actions/runs/37739736920)는
`f6da7970e71edce41efebb7b43a354cb8ad6a14a`의 Linux·Windows·CPU bindings·model CPU
네 job이 모두 성공했다. model suite 426개에는 선택적 empty 관측 수정과 공개 whole
11개가 포함되며, 고정 graphbody의 synthetic Rust 검사 12개도 두 OS에서 통과했다.
이 결과를 아직 연결하지 않은 resident CUDA owner나 실제 Python artifact의 Rust
interop 성공으로 소급하지 않는다. 이번 사용자가 GPU 검증을 보류했으므로 실제
장치 수치·Run·물리 완료·메모리·대국 검사는 미실행으로 유지한다.

### Python packing 산출물과 Rust 검사기의 정적 연결

`device_packing_graph_check` 예제는 `onnx,contracts` feature에서만 빌드한다.
절대 manifest/graph 경로와 독립 SHA-256 두 개를 받아 512 KiB/2 MiB의 고정
입수 상한, regular file·link/reparse·전후 metadata·실제 byte hash를 검사한다.
immutable admission 뒤 원래 읽기 Vec 두 개를 해제하고 고정 body 검사기에 소유권을
넘긴다. report의 session/metadata 각 1-byte 값은 정적 선언 sentinel이며 실제 native
할당·provider·수치·Run·fence·VRAM·등록 출처 인증을 만들지 않는다. path 검사는
일반적인 hostile filesystem의 inode 인증이나 전체 구간의 원자적 path 인증이 아니다.

기존 bounded CPU validation은 독립 Python producer가 만든 실제 artifact를 이 Rust
예제에 전달한다. 모델 수치 검사와 구분한 stage·source/asset pin·exit status를 남기고
기존 managed output·time/cleanup 정책을 유지한다. 두 독립 소스 리뷰와 Rust 파일
서식을 마쳤으며, 실제 interoperability·CLI fixture·양 OS compile은 새 commit의
CPU CI로 별도 인수한다. 이 연결은 ORT Session을 만들거나 GPU를 실행하지 않는다.

해당 연결의 [CI 37741108960](https://github.com/daejunnom/RoveZero/actions/runs/37741108960)는
`80e5256294d13f65997d1531632ba2cbdb842418`의 네 CPU job이 모두 성공했다.
실제 Python producer 산출물을 Rust 예제가 입수한 stage와 반환 scope는 원래
model job 로그로 인수하며, 장치·ORT Session·Run·fence 미실행 범위를 유지한다.

### 전략 V의 실제 action 입력 경계

별도 `strategic_verifier_query.py`는 strict current/frozen dataset과 checked Rules
semantic 입력에서 immutable Query/2를 만든다. 한 query의 실제 branch/task/profile·
자원 controls와 순서 있는 prior를 결합하고, hash·profile 이름·request ID·catalogue
slot·budget bucket을 수치 feature로 바꾸지 않는다. Defer에는 CPU node 예약과
controls를 0으로 두고 profile 차이로 가짜 대안을 만들지 않는다. 원래 등록 자료는
보존한다. canceled/partial/failed CPU deadline receipt는 unknown 관측으로 유지하며
완료 gain validator의 성공 조건을 완화하지 않는다.

메타데이터 canonical JSON의 UTF-8·escape·구두점·반복 참조 크기는 직렬화 전에
남은 byte credit으로 검사한다. prior의 원래 launch bytes/pin 중복과 검증된 JSON
내용의 canonical digest 중복을 모두 거부해 request/launch whitespace·key-order
별칭으로 같은 실행을 중복 관측하지 않는다. digest는 재표기 판별용이며 독립적인
process 실행 증거를 발급하지 않는다. declared byte credit은 전체 Python heap peak가 아니다.

두 독립 source 리뷰와 AST 검사를 마쳤으며 집중 source test methods 43개는 새
commit의 CPU CI에서 별도 인수한다. Query/2는 scoring·utility·target·dispatch·학습·
제품 권한이 모두 false인 첫 입력 단위다. 실제 frozen action scorer, 같은 conditional
witness의 전체 action 완료·인과·비용 인수, 별도 masked target consumer는 후속 필수
단위다. depth 증가·점수 일치·counterexample 부재를 utility로 간주하지 않는다.

### Query/2 fixture 실패 보존과 resident CUDA owner 연결 초안

[CI 37741985878](https://github.com/daejunnom/RoveZero/actions/runs/37741985878)는
`0faf70c3e28436f3ecb36ec3484f074a5248c7b1`에서 Linux·Windows·CPU bindings가
성공하고 model suite 469개 중 1개가 error로 종료했다. Defer 중복 검사의 두 번째
fixture가 원래 관측 receipt보다 짧은 1ms deadline을 선언하여, 중복 검사를 하기
전에 strict receipt 검증에서 거부됐다. 원래 실패 자료를 보존하고 fixture의 두 번째
예산을 99999 nodes/9999ms로 수정했다. 첫 action의 100000 nodes/10000ms와는
계속 다른 controls이며, 제품 Query/2와 완료 receipt의 검사 조건은 바꾸지 않았다.
수정된 fixture의 실제 통과 여부는 후속 exact commit의 CPU CI로 별도 확인한다.

명시적인 CUDA resident owner 초안은 기존 `device_public_memory`·host pages·Warm과
분리한다. 실제 model/manifest/public graph/Rules-composite encoding·전체 runtime
bundle·device·game namespace를 대조하고, 고정 packing graph의 실제 Session·loaded
interface·all-CUDA 초기 배치를 별도로 확인한다. 초기 registry를 만든 뒤 실제 예약
capacity도 같은 aggregate limit에 합산하여 packing Session 시작 전에 거부할 수 있다.
이 계산은 알려진 예약 선언이며 실제 RAM·VRAM peak가 아니다.

실행 수명은 public subset Run/sync → packing Run/sync → finite scalar·mask·device
검사 → private completion capability → 선택 slice publication → 기존 private role
순서다. 논리 취소와 물리 완료를 구분하며 unknown completion·unwind에서는 최초
원인과 session·backing·binding·allocator·pin을 함께 보존한다. 기존 model residency는
loaded model의 session만 센다. 별도 snapshot의 `packing_native_sessions`는 실제
보유한 보조 Session 수이며, 후속 receipt·총 자원 예산에서는 두 관측을 합산해야 한다.
snapshot이나 정적 graphbody 성공만으로 Run·finite·fence 권한을 만들지 않는다.

이 단위는 두 독립 소스 리뷰와 중앙 Rust 서식을 거친 구현 초안이다. 실제 Rust
compile·CPU fixture 인수는 새 CI에 남아 있고, UCI factory·CLI 선택·arena receipt
연결도 후속 단위다. 이번 세션의 GPU 검증 보류를 유지하며 실제 장치·수치·Run·
완료·메모리·대국·학습 검사는 수행하지 않는다.

resident 초안 `184222c25227d66cc2404573c3b7bfa7b6092914`의
[CI 37744387094](https://github.com/daejunnom/RoveZero/actions/runs/37744387094)는
CPU bindings와 model suite 469개가 성공했다. 따라서 앞선 Query fixture 수정은
실제 CPU 검사에서 인수했다. Linux·Windows Rust job은 새 native 파일의 API 연결
오류 10개로 compile 단계에서 실패했다. ORT 전용 오류 함수에 I/O·JSON 오류를
전달한 곳, fallible graph descriptor 문자열을 바로 사용한 곳, 별도 Warm constructor의
새 optional 필드 초기화 누락을 수정한다. 각 오류의 원래 stage·cause와 fail-closed
조건을 유지하고, invalid descriptor 문자열은 오류 또는 shape 불일치로 거부한다.
Warm CPU 경로에는 resident owner `None`만 추가한다. 원래 양 OS compile 실패
자료를 보존하며, 수정 후 실제 Rust 통과는 후속 exact-SHA CI에서 별도 확인한다.

후속 `714859c8b4bdeda5f611fe751625ee862b70d807`의
[CI 37745167614](https://github.com/daejunnom/RoveZero/actions/runs/37745167614)도
model suite 469개와 CPU bindings가 성공했다. 양 OS Rust는 `rz-uci`의 기존 exhaustive
match 세 곳에서 새 metadata command/result를 아직 처리하지 않아 실패했다.
Evaluate·NewGame 응답의 관측 marker를 NN 성공으로 받아들이지 않고 typed 오류로
거부하며, 실제 snapshot ACK는 선택된 유한 관측 경로에서만 소비하도록 연결한다.
이 실패와 뒤의 selected factory 구현을 구분하고, 전체 Rust 성공을 소급하지 않는다.

### 학습 없이 호출하는 frozen action scorer

`FrozenStrategicVerifierScorer`는 exact Query/2를 기존 frozen CPU FP32 validator에
연결하는 별도 offline seam이다. sealed legacy query16과 원문은 그대로 보존하고
새 호출의 NN query seed만 0으로 둔다. 실제 action controls와 순서가 검증된 prior의
known/unknown·비용·PV를 4483개의 고정 private token으로 표현한다. hash·profile 이름·
catalogue slot·budget bucket·raw CPU 점수는 numeric feature로 사용하지 않는다.

1~8개 전체 catalogue의 실제 batch padding·입력·K/V·GQA 반복·attention/FFN scratch·
검증 workspace·분석적 matmul FLOPs를 첫 tensor collation 전에 예약한다. 반환까지
원래 absolute deadline·actual parameter digest·CPU FP32·eval·gradient flags·module
config·checked parent를 다시 검사한다. 동시 kernel의 선점, process RSS나 BLAS allocator
hard cap, 학습된 utility 또는 기력 개선은 주장하지 않는다. 기존 P/C·checkpoint·export·
parameter·Warm·제품 V는 바꾸지 않는다.

독립 리뷰와 총괄 원문·AST 검사를 마쳤으며 source tests 27개를 준비했다. 실제 model을
호출하는 fixture는 합성 QueryFixture의 두 action이고, 전체 8개 catalogue 및 실패
경계 fixture는 numeric stub을 명시해 사용한다. 이번 소스의 실제 실행 인수는 다음
CPU CI에 남는다. 같은 conditional witness의 실제 whole-action 완료·인과·비용과
별도 masked utility target은 아직 후속 필수 단위다.

### 명시적인 resident 선택의 CLI·소유자·arena 연결

새 lane은 `--pals-cuda-record-pages=registered-packing-v1`과 packing manifest·graph의
두 경로·두 SHA 및 inline resource JSON을 함께 받는다. 모든 옵션을 생략한 기존
경로는 그대로 유지한다. 부분 선택·중복·unknown·CPU/mock·외부 helper·미지원 feature는
모델 로딩 전에 거부한다. 기존 arena의 `--pals-device-public-memory=false`는 허용하되
`true`, host pages 또는 Warm 선언과의 혼용은 거부한다. 이 선택은 현재 Linux CUDA
OwnCPU/Fresh·Shared P/C·FP32·단일 물리 worker 범위이며 새 기본값이 아니다.

자원 선언 DTO는 기본 CPU 빌드에서도 읽을 수 있게 `rz-eval`에 분리한다. 누락·null·
unknown·중복·float·overflow와 서로 다른 packing session 선언을 거부하고, 실제 native
타입 변환은 기존 feature 아래에서만 제공한다. 모델의 두 Session과 보조 packing
Session 하나를 따로 센다. 선택 owner의 전체 invocation 상한에는 session·bank·join·
input·metadata가 포함되므로 기존 native device 요청/session 예약을 다시 더하지 않는다.
별도 UCI 직렬화 요청·runtime bookkeeping의 host 예약은 유지한다. 이 값들은
알려진 예약 선언이며 실제 RSS·cgroup·VRAM peak의 관측이 아니다.

독립 검토에서 보조 packing Session이 모델 Session의 arena 설정을 상속하여 별도
packing 선언과 달라지는 결함을 발견했다. packing 선언의 nonzero·정수 범위·등록
artifact 일치를 확인하고 그 값을 실제 packing provider의 `with_memory_limit`에
전달하도록 수정한다. 모델 public/private 선언은 모델 provider의 원래 arena와
정확히 대조하며 직접 backend 설치도 같은 검사를 거친다. 설정을 자동 축소하거나
선언을 관측 peak로 해석하지 않는다. 별도 모델 2GiB/packing 128MiB와 잘못된 선언을
대조하는 순수 CPU fixture를 추가했으며 실행 인수는 최종 CI에서 확인한다.

guarded artifact loader는 정적 CLI와 제품 경로가 함께 사용한다. caller의 자원 선언과
inspection budget으로 검증한 owned graph만 배포 생성자로 이동한다. 실제 process
epoch·모델·Rules encoding·전체 CUDA runtime bundle·device namespace는 loaded backend와
owner에서 얻는다. 별도 implementation SHA는 추적 소스의 출처이며 모델 의미·Run·
finite·fence 권한을 대신하지 않는다. arena는 기대 implementation 핀을 실제 marker와
대조하고, startup와 종료의 관측은 각각 worker의 실제 metadata ACK로 받는다. 초기
snapshot을 종료 결과로 복사하지 않는다. 관측 실패·unknown completion·격리 상태는
성공으로 채우지 않으며 packing 작업 수를 모델 NN 입력 수나 탐색 소비 수와 섞지 않는다.

독립 CLI·loader 소스 리뷰와 중앙 소스 검토를 수행했지만 이 연결 단위의 최종 Rust
compile·fixture·정적 interoperability 인수는 다음 정확한 SHA의 CPU CI 대상이다.
사용자의 이번 GPU 검증 보류를 유지한다. CUDA 장치·수치·물리 완료·메모리·성능·대국은
미실행이며 실제 training·backward·optimizer와 새 cloud 실행도 수행하지 않는다.

### 선택 연결의 CPU CI 실패 보존과 utility 준비 경계

통합 SHA `5a85621a7b5d805da2f6ac53d78722f3e165399b`의 run `37749160614`는 완료 실패다.
CPU bindings job만 성공했고 Linux·Windows Rust job은 `pals_model_check` 예제의 두
exhaustive match에서 새 metadata 응답을 빠뜨려 compile 단계에서 실패했다. 예제도
해당 응답을 평가·새 게임 성공으로 처리하지 않고 명시적인 오류로 거부하도록 수정한다.
모델 job은 470 tests 중 scorer module import 한 건이 실패했다. parameter digest의
실제 정의인 `preparation_check`로 테스트 import를 수정한다. 이 실패에서 scorer의
27개 test가 실행·통과했다고 기록하지 않는다. 세 실패 job 원문과 SHA를 외부 보고서
루트에 보존하며 이후 정확한 통합 SHA의 전체 CPU CI로 다시 확인한다.

`strategic_verifier_utility`는 Query/2·criterion·before pair·각 action의 CPU 원문과
핀·whole witness의 재검증 접점을 준비한다. immutable raw와 expected-pin metadata의
합계는 parse·canonicalization 전에 4MiB credit으로 제한한다. 이는 해당 원문의 상한이며
전체 Python heap·기존 자산·JSON workspace의 process peak 한도가 아니다. 독립 리뷰에서
caller bundle의 unknown-key set 복제가 byte guard보다 먼저 발생하는 문제를 발견하여
exact dict·길이 선행·고정 이름 membership 순서로 거부하도록 수정했다.

현재 반환은 항상 `masked_unresolved`·`native_action_causal_bridge_unobserved`이며
`known=false`·`mask=false`·`actual_utility_groups=0`이다. preference·sign·whole action cost는
만들지 않고 utility·target·training·action completion·final closure 권한은 false다.
CPU response 수치나 caller before 선언을 whole-action 인과·실행 비용으로 승격하지 않는다.
기존 coverage utility와 별개이며 scorer·Pareto·loss·학습 소비자에 아직 연결하지 않는다.
24개 focused test source와 독립 리뷰·AST parse를 준비했다. 실제 CPU CI 실행 및 실제
native action causal bridge·whole-action cost·utility target 구현은 별도로 남아 있다.

`dc64d63c4a8fe6ba282c6a4c25b4ab8f2f4e2ecb` / run `37750409269`에서는 model CPU의
520개 tests와 bindings가 실제 통과했다. scorer 27개와 utility 24개는 이 run의 실제
성공이며 합성 action/witness 범위와 실제 학습·utility group 부재는 그대로 유지한다.
Linux·Windows의 workspace all-targets/all-features, native CLI, 독립 Rules release와
CPU Python 검사도 통과했으나 두 OS job의 최종 상태는 Clippy 한 건으로 FAILURE다.

Clippy는 resident의 `Source::Hit(DeviceProjectionOffset)`와 `Pending(u16)`의 크기 차이를
지적했다. 이 값은 board 1개·record 최대 128개의 inline Copy port로 제한되며
`ActiveCudaPageInvocation` 전체 크기를 실제 metadata 예약에 더한다. 해당 enum에만
설명과 `large_enum_variant` 예외를 두어 요청별 heap·새 fallible 소유 경계를 추가하지
않는다. 의미·메모리 효율·GPU 성공의 개선으로 보고하지 않는다. Linux all-features에서는
미지원 빌드 fixture가 cfg로 제외되므로 기존 단일 CI의 두 OS job에 default-feature CLI
unit 검사를 추가하여 provider·artifact 로딩 이전 거부를 별도로 실행한다. 후속 SHA의
Clippy와 이 기본 feature 검사는 다시 인수한다.

`b1715a23eb78a17eda3be0f813d24961a7b63ecf` / run `37751846236`의 model 520 tests와
bindings는 통과했다. 양 OS의 workspace tests·새 default-feature binary CLI 검사·
native CLI·Rules release·보조 CPU 검사는 성공했으나 최종 Clippy에서 arena의
`PalsEndpointLaunchV3` 크기 차이가 추가로 드러나 두 job은 FAILURE다. 앞선 native
타입 예외는 해당 crate의 Clippy를 통과했고 이 결과를 arena 전체 성공으로 확대하지 않는다.

arena의 새 optional `cuda_record_pages` 설정만 `Option<Box<...>>`로 보관하여 선택하지
않은 endpoint가 큰 resource DTO를 inline으로 갖지 않도록 한다. native 요청 port와
다르게 이 값은 시작 때 선택한 설정 하나이며 요청별 allocator·물리 수명을 바꾸지 않는다.
Box는 JSON에 투명하고 기존 present-value deserializer의 명시적 null 거부·unknown
거부·None 생략·canonical digest는 유지한다. 기존 wire·launch·audit fixture로 다시
검사하며 실제 process peak·성능 효과의 계측 증거는 아니다. 원래 실패 로그를 보존한다.

### Query/2와 실제 CPU_T 요청의 사전 결합

`strategic_cpu_action`은 기존 `CheckedStrategicQuery`를 재검증한 뒤 catalogue의 선택
행, exact semantic input, 등록 프로필, 부모 Rules 상태·이력, CPU 원문 요청과 독립 핀을
결합한다. 결과를 보기 전에 envelope를 고정하며 기존 private CPU 요청을 임의로 수정하거나
남은 시간에 맞춰 `max_wall_time_ms`를 줄이지 않는다. 요청·envelope·출력은 별도 schema와
원문 핀으로 구분한다. Python factory는 파일을 읽거나 프로세스·모델·학습을 실행하지 않는다.

Rust의 `pals_cpu_task::strategic_action`과 예제의 명시적 `--strategic-action` 선택은
`rz-pals-private-strategic-action/1` envelope를 strict typed serde로 읽는다. 실행 전에
선택 action과 실제 CPU 요청의 task·깊이·노드·wall·출력·등록 프로필을 대조하고, 남은
steps·nodes·wall·출력 예약이 충분한지 확인한다. CPU 검사 action은 최대 두 번의 node
예약을, Defer는 node 0을 사용하며 출력 예약은 action의 두 배다. 기존 원래 start instant와
CPU_T dispatcher를 재사용하고 기존 legacy 요청·조건·CLI의 의미는 바꾸지 않는다.

이 Rust 경계는 **caller가 검증·등록한 Query/2 선언을 실제 own CPU dispatch에 결합**한다.
전체 Query/2의 current parent·frozen admission·semantic capability·prior chronology를
Rust가 다시 검증했다고 주장하지 않는다. `query_revalidated_by_rust=false`를 보존하고
그 검증은 원문·독립 핀을 보유한 strict Python consumer가 담당한다. profile·binary의 선언은
actual loaded image·source·process의 독립 관측을 대신하지 않는다. CLI는 자신의 loaded
image 또는 current-exe-path 확인 범위를 별도로 표시하며 다른 host의 보장으로 바꾸지 않는다.

Rust 응답에는 실제 CPU 결과 원문과 byte 핀을 남긴다. 실패하면 원래 `CpuTaskError`의
baseline·after·known/unknown work를 보존한다. 응답 직렬화·deadline·출력 전달 실패도
진단 자료이며 성공한 action이나 무비용 결과가 아니다. CPU의 새 조건부 질문 완료는
Rules 판정·native Reply/Repair/recheck·publication·전체 search closure와 별도 증거다.

이번 단계로 native action 인과, 같은 conditional witness의 독립 두 action 비교, whole-action
비용 및 학습 utility가 완성되지 않는다. `native_action_causal_bridge_observed`,
`conditional_witness_validated`, `whole_action_cost_observed`, `final_search_closure_observed`,
`action_completion_admitted`와 utility·target·training authority는 false다. 기존
`strategic_verifier_utility`의 항상 masked 반환과 `actual_utility_groups=0`도 유지한다.
제품 V-free 경로, actual training·backward·optimizer 제외 및 이번 GPU 검증 보류는 그대로다.

새 source의 CPU 검사와 기존 소비자 호환성은 후속 정확한 commit SHA의 동일 CPU CI에서
별도로 인수한다. source 검사·합성 caller fixture를 실제 CUDA·native action·대국의 성공으로
기록하지 않는다. 후속 native producer는 task 선택 전부터 actual native request ID·task
execution·최종 return/cleanup·단일 clock의 비용을 결합해야 하며 사후 digest 첨부로 대체하지 않는다.

### CPU action 경계의 정확한 CI 인수와 native 반환 관측

`1613c75ed64b0511ad1eb113722d43367e4959dd`의
[CPU CI run 37758023927](https://github.com/daejunnom/RoveZero/actions/runs/37758023927)은
Linux·Windows·모델 CPU·bindings 네 job이 모두 성공했다. 후속 Python 준비·소비를 포함한
`f78b194220cc6bb0358bd95c16e3510f572d9317`의
[CPU CI run 37758783207](https://github.com/daejunnom/RoveZero/actions/runs/37758783207)도
동일한 네 job이 모두 성공했다. 총괄은 최신 run의 원문에서 모델 연구 suite 548개와
새 Python focused 28개, 양 OS 각각 Rust 전략 경계 11개의 실제 통과를 확인했다.
Python receipt는 합성 fixture이며 Rust의 작은 자체 CPU primitives는 CI에서 실행된다.
네 job의 원문·byte 수·SHA와 정확한 실행 HEAD를 소스 밖에 보존한다. 이 결과를 실제
native action·whole cost·CUDA·기력·학습의 인수로 확대하지 않는다.

후속 native 관측은 실제 collector의 `engine.search` 호출 전과 반환 후를 소유한
[`native.rs`](../../crates/rz-arena/src/pals_collect/native.rs)에 둔다. 역사적 Query/2의 CPU
결과를 별도 full search에 붙이는 방식은 인과 연결로 인정하지 않는다. Query/2의 strict
parent는 완료된 collection의 독립 receipt·raw audit·current view·frozen admission을
요구하므로 진행 중인 C/Repair row를 같은 invocation의 Query/2 parent로 받을 수 없다.
첫 action 구현은 완료된 frozen parent를 기존 strict loader로 인수하고, 새 Rust owner가
Rules 상태·전체 이력·prefix·restriction을 재구성하는 명시적 offline replay 범위로 둔다.
과거 parent의 RequestId·epoch·Proposal·Repair는 provenance이며 새 수락·publication
권한으로 복사하지 않는다. 선택 CPU 검사와 새 실제 Proposal·Reply·Repair·recheck는
같은 fresh invocation의 TaskKey·ExecutionId·ObservationId로 이어야 한다. live parent
continuation에는 별도 불변 capability와 pending·reset·종료 계약이 필요하며 기존 frozen
조건을 느슨하게 만들지 않는다. 실제 replay action API와 독립적인 두 action의 동일
조건부 fact·전체 비용 비교는 아직 구현·인수 대상이다. utility의 masked 상태와 group 0을 유지한다.

별도 `rz-pals-native-observed-search-return/1` 자료는 등록된 explicit recheck lane에서만
검색 전 descriptor와 실제 검색 반환을 관측하는 경계다. 기존 default·V1·recheck witness의
wire와 의미는 유지한다. 완전 Rules replay의 origin·completeness·모든 알려진 이동을
`PositionSnapshot::uci_replay(4096)`으로 보존하며 FEN의 unknown prefix를 완전 이력으로
승격하지 않는다. 두 행은 각각 newline 포함 64KiB, 기존 요약은 8KiB로 제한하여 총
139,264 bytes와 세 행을 검색 전에 함께 예약한다. 남은 output에 맞추어 의미 필드·이력·
deadline을 조용히 축소하지 않는다. 메모리 seal은 디스크 write/fsync 완료 증거가 아니다.

실제 Result·counters의 존재 여부·원래 오류·마지막 cancel/deadline 상태를 보존한다.
검색 반환 직후의 시각과 이후 owner snapshot·직렬화 시각은 동일 sink clock 안에서
구분하며 caller 전체 wall clock을 대신하지 않는다. owner receipt의 순차 lock/Acquire
관측은 원자적 ledger나 새 worker Stats ACK가 아니다. physical NN rows는 unknown이며
role call 수로 환산하지 않는다. Query action 인과·whole-action cost·utility·학습 권한은
false이고 search return은 worker join·session/buffer 해제와 별도다. 종료 행·요약·회수
실패는 원래 search primary 뒤에 secondary로 보존하며 관측 work를 지우지 않는다.
이 후속 변경의 compile·fixture·실제 collector 인수는 원래 f78b194의 성공과 구분한다.
첫 WIP 단위는 output artifact 허용과 합성 persistence·sibling failure 검사였다. 아래
89eea8c의 성공과 후속 실제 호출 경계의 생산자 구현·검사 결과를 별도로 기록한다.
사용자의 이번 GPU 검증 보류와 actual training·backward·optimizer 제외는 계속 적용한다.

output persistence WIP `89eea8c6f4d3987882ddaa5925c183dbb1e794c1`의
[CPU CI run 37762510301](https://github.com/daejunnom/RoveZero/actions/runs/37762510301)은
네 job이 모두 성공했다. 양 OS 원시 로그에서 새 persistence 합성 fixture 두 개, 모델
suite 548개·Python 전략 경계 28개·Rust 전략 경계 11개/OS의 통과를 확인했다. 이 HEAD에는
실제 검색 전 예약·반환 생산자가 없으며, 자료 보존 fixture 성공을 native action 인과나
whole cost 인수로 바꾸지 않는다. 정확한 HEAD·job·원문 pin은 소스 밖 인수 기록에 보존한다.

후속 Rust 생산자는 explicit recheck의 실제 engine getter와 등록된 typed policy/source를
dispatch 전에 대조하고, descriptor·return·summary의 전용 credit을 함께 예약한다.
실제 `engine.search` 바로 뒤에 반환 시각·cancel을 고정한 뒤 counters의 `Option`,
순차 owner snapshot, 직렬화 시작을 기록한다. 오류 뒤에도 summary 저장을 시도하고
trace 회수의 missing physical 진단은 원래 search primary 뒤에 보존한다. 기존 Disabled
summary 경로는 유지한다. 추가한 일곱 focused fixture는 합성 owner JSON·제공된 오류로
예약·슬롯·이력 상한·unknown·시계·오류 보존을 검사하며 실제 engine/CPU/NN 인과 자료를
생산하지 않는다. 소스 구현·정확한 파일 formatter·독립 검토·해당 HEAD의 CPU CI와
등록된 실제 collector 실행을 각각 구분한다. Query-selected replay action API와 전체
비용 ledger는 이 반환 생산자 구현만으로 완료되지 않는다.

첫 생산자 공유 HEAD `ec0fba3b02a9c9b845e7a8647a2dc749978dec89`의
[CPU CI run 37764947833](https://github.com/daejunnom/RoveZero/actions/runs/37764947833)은
실패했다. Linux·Windows는 44-field 단일 JSON 매크로의 재귀 한도에서 컴파일을
중단했고, 모델 CPU suite는 변경한 Rust source에 대한 strict 등록·독립 fixture pin이
갱신되지 않아 52 failure·3 error를 보존했다. bindings job만 성공했다. 후속 수정은
카운터 44개를 그대로 개별 scalar JSON 객체로 조립하며 crate 재귀 한도는 변경하지
않는다. source 등록은 독립 검토한 정확한 collector/engine pair를 명시 literal로 추가하고
이전 pair와 실패 자료를 보존한다. 실행 시 current hash를 자동 허용하거나 unknown
source를 인수하지 않는다. 수정 HEAD의 CI 결과는 이전 성공·실패와 별도로 확인한다.

`c9ed3a6933ab37bb3f0ed5770d3482dfc34962e4`의
[CPU CI run 37766471341](https://github.com/daejunnom/RoveZero/actions/runs/37766471341)은
모델 CPU와 bindings job이 성공했지만 전체 run은 실패했다. Windows는 clock fixture의
1ns subtraction이 동일 `Instant`로 관측되어 before-origin assertion에서 실패했고,
Linux는 해당 native focused 검사들을 통과한 뒤 `collapsible_if`의 warnings-as-errors에서
중단했다. 후속 변경은 fixture 차이를 1ms로 만들고, Err → sink lock → selected 여부의
평가 순서를 유지한 let-chain으로 조건문을 정리한다. production deadline이나 lint
엄격도는 바꾸지 않는다. 최종 source pair를 명시 등록하고 같은 HEAD의 재검사를 요구한다.

최종 생산자 수정 HEAD `b738ad719c877d0fe2b195d5755ddb44f22c6a43`의
[CPU CI run 37767689597](https://github.com/daejunnom/RoveZero/actions/runs/37767689597)은
Linux·Windows·bindings·모델 CPU 네 job 모두 성공했다. 원문 로그에서 모델 suite 548개,
Python 전략 경계 28개, Rust 전략 경계 11개/OS와 native fixture 29개/OS를 확인했다.
일곱 새 반환 fixture는 합성 owner 자료를 사용하는 범위이며 실제 collector 실행,
Query 선택의 native 인과, whole cost, utility를 인증하지 않는다. 이전 실패 자료도 보존한다.

### Frozen parent의 별도 Rust replay 접점

[engine/replay.rs](../../crates/rz-search/src/pals/engine/replay.rs)의
`FreshReplayOwner::new`와 일회성 `run`은 별도의 caller-declared offline 경계다.
첫 범위는 Complete startpos 전체 이력의 Rules 재구성 → 실제 새 P Proposal →
선택 `defend_response`의 두 CPU 검사 → after 관측의 공개 Counterexample →
같은 replay의 실제 Reply 수락까지다. 기존 `engine.rs`에는 모듈 선언만 추가하며
기존 product search·CPU dispatcher·native collector 함수 본문은 유지한다.

- 역사적 parent·Query·catalogue·before-result·semantic·CPU request digest는 선언된
  provenance다. strict Python admission과 원문 byte binding은 호출자의 인수 책임이며
  과거 record·TaskKey·ExecutionId·ObservationId를 새 stores로 가져오지 않는다.
- 새 P 출력이 선언 prefix와 맞지 않으면 `SeedNotApplied`로 끝난다. 법적 historical
  claimed line도 가설로 보존하며 CPU가 그 전체 수순을 검증했다고 표시하지 않는다.
- baseline H0와 after H1은 동일 H1 capability의 새 PlanAssisted CPU/TT로 각각 수행한다.
  같은 effective Rules root order와 고정 N, 동일 original deadline/cancel을 사용하고
  2N 지원이 부족하면 실행 전에 거절한다. 원래 restriction 순서는 별도로 보존한다.
- phase별 TaskKey에서 Start만 허용한다. 원문 report·attempt·known/unknown work를
  CPU 교체 전에 소유하고 primary와 secondary task cleanup 실패를 구분한다.
  정확한 DepthLimit·요청 깊이·CompletedIteration·reuse 0을 충족한 after만 게시한다.
- 공개 Counterexample의 값은 None·depth 0이며 실제 after observation을 참조한다.
  그 이후 새 Reply context를 준비·수락하고 늦은 취소·만료를 다시 검사한다.
  같은 Proposal 응수는 `SameProposalResponse`, 미완료 검사는 `Partial`로 보존한다.
- entered invocation마다 논리 `finish_search` hook를 한 번 호출한다. 이 hook와
  Reply 수락은 worker join·buffer 해제·Repair·recheck 완료의 증거와 구분한다.
  모델의 native worker/epoch freshness도 이 owner의 Rust 생성만으로 인증하지 않는다.

13개 focused CPU fixture는 실제 own CPU 실행과 scripted RoleModel을 연결하여 phase
독립성·Start-only·H0 완료 후 H1 partial의 work 보존·Reply 입력·late cancel·finish를
검사한다. 소스/formatter/독립 검토와 해당 HEAD의 CI 성공을 구분하며, 아래 후속
검사 결과를 확인하기 전에는 새 fixture가 통과했다고 표시하지 않는다.
기존 source pair의 successor 등록은 unchanged default/recheck 경로에만 적용되며
새 replay를 기존 witness 권한으로 자동 인수하지 않는다.

생성자와 Rules 재구성 비용은 `run` elapsed 이전이므로 전체 invocation 시간은 아직
미관측이다. 실제 Query/2 소비자·native RequestId/NN 실행·같은 invocation의 Repair와
recheck·caller 종료/정리 ledger 연결은 후속 구현·인수 범위다. 독립 두 action의 동일
조건부 fact/전체 비용 비교는 아직 미인수이며 utility groups 0/masked를 유지한다.
이번 GPU 검증 보류와 actual training/backward/optimizer 제외를 유지한다.

이 첫 replay 접점의 HEAD `2f4e4ffbe9ac7dd3602f599307dd0530b085c83e`에서
[CPU CI run 37772249341](https://github.com/daejunnom/RoveZero/actions/runs/37772249341)은
Linux·Windows·모델 CPU·bindings 네 job 모두 성공했다. 총괄은 원문에서 새 replay
fixture 13개/OS, 모델 suite 548개·Python 전략 경계 28개·Rust 전략 경계 11개/OS,
native fixture 29개/OS·output persistence 2개/OS의 실제 통과를 확인했다. 정확한
HEAD·job 종료 시각·원문 byte 수·SHA는 저장소 밖 인수 자료로 보존한다. 이 결과는
작은 자체 CPU와 scripted RoleModel의 연결 검사이며 실제 Query/2 인수·native
request/NN 인과·Repair/recheck·whole cost·GPU·기력·학습 인수로 확대하지 않는다.

### 명시적인 Repair endpoint replay

별도 `run_with_repair`는 기존 `run`의 Reply-only 2N 의미를 보존하면서 실제 Reply
수락 뒤의 모델 continuation·Repair와 독립 endpoint 검사를 연결한다. 시작 전에
고정 3N, 최대 역할 호출 `L + 1 + 2 × (L − P − 1)`, 네 public record와 세 stage,
유한 node·observation·line 저장 한도를 확인한다. `L`은 최대 line 길이, `P`는
고정 prefix 길이다. 남은 예산에 맞추어 N·깊이·수순 길이를 조용히 낮추지 않는다.

새 C 수순은 실제 선택 Reply response의 Rules 상태에서 생성한다. 그 응수가 H1
CPU PV와 다르면 CPU의 suffix를 이어 붙이지 않는다. 모델이 생성한 full
Counterexample과 Repair record는 CPU 출처 None·값 None·depth 0·scope None으로
게시한다. 원래 H1 observation은 초기 CPU Counterexample과 최초 Reply context의
출처로 남으며 새 모델 수순 전체를 검증한 근거가 아니다. 실제 Repair 수락 수 증가와
전체 Rules 수순·끝 상태·합법 수 순서의 재대조 없이 완료 endpoint를 주장하지 않는다.

세 번째 CPU/TT는 새 owner이며 질문은 `AnalyzePosition`, root moves는 빈 목록,
input revision은 0이다. 별도 `rz-pals-frozen-parent-repair-endpoint-replay/1` scope의
Start-only H1/N Task와 원문 report·attempt를 보존한다. 정확한 완료 관측·Task 소비가
끝난 뒤에만 endpoint의 own CPU evidence를 설치하고 기존 `recheck_endpoint`의
state·execution·question·scope provenance 검사를 통과해야 한다. Partial은 관측을
남기지만 완료 Node evidence를 만들지 않는다. Rules terminal은 세 번째 CPU 실행
없이 Rules 근거의 별도 결과로 반환하며 CPU mate 점수를 종료 판정으로 승격하지 않는다.

추가한 12개 fixture source는 작은 실제 own CPU와 scripted Repair 모델을 사용한다.
기존 13개 fixture 및 Reply-only `run_inner`·CPU stage 본문은 유지하고, one-shot 사용·
원래 deadline/cancel·primary/secondary 실패·finish 한 번은 공통 wrapper에서 보존한다.
소스 대조·formatter·독립 리뷰와 새 정확한 HEAD의 CPU CI 결과는 별도로 인수한다.
앞선 2f4e4ff의 13개 통과를 새 12개 검사 성공으로 재사용하지 않는다.

이 경계는 accepted Repair의 끝 상태 관측까지다. Supported Repair·조건부 refutation·
native post-Repair recheck·전체 caller 비용·물리 종료·utility는 아직 인수하지 않는다.
후속 wire는 별도 request/observation 및 source·provider factory·실제 실행 binary 등록이
필요하다. 역사 CPU binary pin을 새 replay binary pin으로 대체하거나 새 결과를 기존
2N action receipt·Query/2 completed prior로 재라벨하지 않는다. 3N mode는 추가 역할·
저장·출력·단일 전체 wall·cleanup 한도를 함께 봉인하고 원래 remaining이 부족하면
거절해야 한다. 기존 utility groups 0/masked·제품 V 미활성·이번 GPU 검증 보류와
actual training/backward/optimizer 제외를 유지한다.

Repair endpoint 초안 HEAD `bbc673b8713914666bd8da1706b367ebe0aa1557`의
[CPU CI 37776614455](https://github.com/daejunnom/RoveZero/actions/runs/37776614455)는
모델 CPU·bindings 성공, Linux·Windows 실패로 종료됐다. 각 OS에서 기존 Reply-only
13개와 새 Repair endpoint 11개는 통과했으나 terminal fixture 한 개가
`Search(Role(InvalidOutput))`로 실패했다. root `f2f3` 뒤 초안의
`g7g5 g2g4 d8h4`는 e7 폰이 퀸의 경로를 막는다. 후속 수정은 fixture prefix와
기대 수순만 `e7e5 g2g4 d8h4`로 바꾸며 제품 Rules·평가 출력 검사는 유지한다.
이 수순은 기존 부모의 Rules terminal fixture와 일치한다. 원시 실패·job별 종료·
이후 skip된 검사를 보존하고 수정 HEAD의 별도 CI 성공 전까지 전체 suite를 인수하지 않는다.

수정 HEAD `af2b24e5bd2440fc15f613e278e794921909c33b`의
[CPU CI 37778289887](https://github.com/daejunnom/RoveZero/actions/runs/37778289887)는
Linux·Windows·모델 CPU·bindings 네 job 모두 성공했다. 원시 로그에서 기존
Reply-only 13개/OS와 Repair endpoint 12개/OS의 통과를 확인했고 이전 실패와 skip
기록은 보존한다. 작은 실제 own CPU와 scripted RoleModel의 연결 인수이며 native
모델·전체 Repair recheck·Query 승인·whole cost·물리 종료·utility·GPU 인수는 아니다.

### 소비자와 실행이 공유하는 순수 Repair 자원 계산

`repair_replay_requirements(L, P, N)`과 읽기 전용 `RepairReplayRequirements`를
공개하여 기존 3N·역할 호출·node·observation·line chunk·record·stage 계산을 한
소유 경계에서 재사용한다. 실제 `prepare_repair`도 이 함수의 반환값으로 비교하며,
private `StoreLimits` 계산과 실제 예약은 탐색 소유 경계에 둔다. 함수는 I/O·할당·
모델·CPU 호출 없이 checked 산술로 잘못된 scalar와 중간 overflow를 거부한다.
이 반환값은 요구량 선언이며 할당 peak·provider 지원·실제 실행 인수는 아니다.
제품 L/N/config 상한·Rules·수명 검사는 기존 constructor와 실행 경로가 담당한다.

기존 25개 replay fixture와 run·Repair·endpoint·CPU stage 본문은 유지한다. 새
순수 공식·유효/잘못된 경계·overflow·실제 preflight 기준 연결 검사 네 개의 결과는
새 정확한 HEAD의 CPU CI로 별도 인수한다. 첫 새 입력 경계는 rz-uci 소비자 쪽에서
원본 prepared action과 전체 SemanticReceipt 원문·독립 pin·등록·네 parent pin을
비교하고 기존 Rules 생성기를 재사용하는 방향이며 아직 구현·실행 인수 전이다.
기존 Query/2·CPU action/1의 원문과 2N·binary 의미를 자동 확장하지 않는다.

이 공유 자원 계산의 HEAD `c7245e1a056f13ca5f490a8fb312f4db415eb2bf`에서
[CPU CI 37781627761](https://github.com/daejunnom/RoveZero/actions/runs/37781627761)은
Linux·Windows·모델 CPU·bindings 네 job 모두 성공했다. 원문 로그에서 각 OS의
기존 Reply-only 13개·Repair endpoint 12개와 새 순수 계산/preflight 네 개의 통과를
확인했다. 모델 suite 548개 중 Python 전략 경계 28개, Rust 전략 경계 11개/OS,
native fixture 29개/OS와 output persistence 2개/OS도 확인했다. 원문 로그와
정확한 HEAD·job 종료·byte 수·SHA는 저장소 밖에 보존한다. 이는 새 rz-uci 입력
경계·실제 등록 자료 수집·strict Query/native recheck·whole cost·물리 종료·utility·
GPU·학습 인수와 구분한다.

### 원본 자료를 유지하는 Rust replay 입력 경계

`rz-uci`의 `strategic_action::replay_inputs`는 별도
`rz-pals-frozen-replay-inputs/1` 요청과
`rz-pals-frozen-replay-consumer-registration/1` 등록을 받고
`check_replay_inputs(bytes, independent_expected_pins, original_started)`로
검사한다. 원본 prepared action·전체 SemanticReceipt·등록 UTF-8 bytes의 길이·SHA와
네 parent pin, Query·catalogue·before·prior·semantic pin을 독립 caller 값과 비교한다.
semantic capability identity·receipt context·branch meaning·before anchor·원문 SHA는
각 도메인으로 유지한다. 과거 CPU binary와 새 replay binary·engine·wrapper·replay
child·provider factory 소스 선언은 별도로 봉인한다. 등록 digest는 loaded image를
실제 관측한 증거가 아니다.

기존 action/CPU decoder·pin 검증과 Rules `prepare_root`·`describe`·수순 재생을
재사용한다. 실제 root와 prefix target, claim 끝 상태의 전체 descriptor·이력·차례·
합법 수 순서·restriction·승격 token을 원본 receipt와 대조한 뒤 private
`CheckedReplayInputs`를 만든다. target-relative claim에는 prefix를 정확히 한 번
붙인다. 이 첫 경계는 완전 startpos 이력의 known positions 최대 4096개를 지원하고
FEN unknown history나 한도 초과를 거절한다. nullable 필드가 원문에서 빠져 serde가
None을 채운 경우도 typed receipt와 raw JSON의 shape 대조로 거절한다.

`reply_only_2n`과 `repair_endpoint_3n`은 별도 모드다. 정확한 2N/3N allowance와
출력·역할·store 선언은 원래 remaining에 들어맞아야 한다. source-owned 순수 자원
계산을 재사용하며 Reply-only의 보수적 Repair role/store 예약은 overreservation으로
기록한다. 전체 deadline은 원래 시작+whole wall이며 owner work deadline은 같은
시작에서 명시 cleanup reserve를 뺀 값이다. invalid output·pin·등록·control 오류에도
유효한 원래 whole wall과 elapsed·진단 cap·상위/하위 원인을 보존한다. 생성자와 Rules
준비 비용을 제외한 search elapsed를 전체 invocation 비용으로 쓰지 않는다.

checked 객체는 원문·등록·자원·두 deadline의 읽기 전용 accessor와 기존 typed owner
입력으로의 consuming conversion만 제공한다. CPU engine·모델·provider·owner를
생성하거나 dispatch하지 않는다. Query/semantic capability·과거 source/launch 인수는
caller 책임이며 Rust 재인수·native 인과·full Repair recheck·전체 비용·물리 종료·
utility·target·training·제품 V 권한은 모두 false다. 기존 native engine/producer의
closed source pair와 Python 등록 literal을 새 child의 증거로 자동 확장하지 않는다.

후속 caller는 기존 strict parent와 선택 index에서 최초 semantic admission의 원본
request·receipt·source·registration·before·launch·common query bytes와 독립 pins를
보존하고 기존 factory로 재대조해야 한다. `rules_receipt()`의 deep-copy dict를
재직렬화한 bytes는 원본 receipt의 대체물이 아니다. prepared action과 같은 선택
capability를 대조한 뒤 원본 전체 receipt와 새 replay 등록을 전달한다. 과거 CPU/
semantic 등록·binary와 새 wrapper build·binary·launch 관측을 각각 보존한다.

독립 소스 검토에서 decoded prefix의 Vec retained capacity가 작은 L보다 커질 수
있는데 실제 owner는 이를 거부한다는 연결 공백을 발견했다. 새 child는 prefix와
claim에 L, root/target legal order와 restriction에 256의 retained capacity 상한을
보장한다. ordered 길이를 checked 합산하고 fallible exact reservation 뒤 실제
capacity를 확인하며 초과는 명시적으로 거부한다. L=2/P=1/no-claim의 두 모드에 대해
반환 owner 인자·원래 원문·자원·no-engine audit을 검사하는 회귀를 추가한다.

두 모드·claim 결합·raw self-repin·encoding/domain 혼동·binary/source 교환·Rules
descriptor 변조·history/길이·3N remaining 부족·자원/cleanup·닫힌 wire·원래 만료
clock·오류/권한·retained capacity 검사의 fixture 13개를 추가했다. 기존 두 조상 파일에는 module 선언과
`prepare_root` visibility만 변경했고 parent engine·native producer·replay는 불변이다.
formatter·소스 대조·독립 리뷰는 실제 새 HEAD CPU CI 실행 결과와 구분한다. 초기
작성 시 실행 인수 전 기록은 아래 정확한 HEAD의 후속 관측으로 보완하며,
utility groups 0/masked·제품 V 미활성·이번 GPU 검증 보류와
actual training/backward/optimizer 제외를 유지한다.

입력 경계 HEAD `f36cc5b5d6cf145fa228eea40200502e752429fc`의
[CPU CI 37796629924](https://github.com/daejunnom/RoveZero/actions/runs/37796629924)는
Linux·Windows·모델 CPU·bindings 네 job 모두 성공했다. 원문에서 각 OS의 새 입력
fixture 13개를 확인했으며 L2/P1 retained capacity 회귀도 포함한다. 기존 Reply-only
13개·Repair endpoint 12개·순수 requirements/preflight 네 개, native fixture 29개와
persistence 두 개의 결과도 별도로 확인했다. 이 결과는 원본 caller 선언과 실제
Rules를 대조하는 경계의 CPU 인수이며 새 native factory/실제 replay dispatch·요청별
물리 lease·strict Query/full recheck·whole cost·physical closure·utility 인수는 아니다.
새 native 관측 변경은 별도 HEAD와 검사로 관리한다. 원문 job 로그·byte 수·SHA와
job별 완료 시각은 저장소 밖에 보존한다.

### 요청별 actual Runtime ExecutionId 관측 접점

`pals_native.rs`의 선택적 `NativeRoleObserver`에 기본 noop인 `dispatched`와
`terminal` 접점을 추가한다. 실제 `worker.submit` 성공 뒤 확보한 Lease의
`RequestId`·`ExecutionId`를 `NativeRoleExecutionBinding`으로 전달한다. receipt의
aggregate high-water를 요청별 실행 ID로 환산하지 않으며 기존 receipt schema와
prepared/physical return/consumption 접점은 유지한다.

제출 뒤 observer 오류·panic·mutex poison이 생겨도 물리 Lease를 `Ok(Lease)`로
runtime에 넘겨 같은 물리 작업을 계속 회수한다. backend 원래 실패는 callback
전에 봉인하고 observer 오류는 별도 secondary로 남긴다. 실제 Ready만 inflight와
물리 완료 계수를 회수하며 observer 실패 결과는 정상 delivery/consumption으로
전달하지 않는다. callback 성공은 물리 fence나 서비스 성공 증거가 아니다.

`NativeRoleTerminal::CompletionUnknown`은 quarantine·consumed·drain 만료를
같은 binding에서 한 번 관측하는 상태이며 물리 완료가 아니다. 별도로 실제 늦은
Ready를 관측해도 quarantine과 닫힌 정상 소비 권한을 복구하지 않는다. pending은
Ready 행을 만들지 않는다. 단일 bounded binding slot은 기존 runtime의 실제
max_requests/max_batch_items/max_executions 각각 1과 맞추고, 기존 Ready 미관측
binding이 있으면 새 worker submit을 거절한다.

실제 ID/high-water/new-game 분리, 제출 후 오류·panic의 Lease 보존,
backend/observer 동시 실패, unknown 중복 방지, drain 뒤 late Ready,
observer mutex poison, 잘못된 control 응답의 완료/성공 분리를 다루는 controlled
fixture 소스 일곱 개를 추가했다. native의 test header는 기존 41개에서 48개가 됐다.
formatter와 기존 입력·replay·runtime·worker·규약·workflow 보존 대조는 완료했고
동결 소스의 독립 리뷰에서 추가 필수 수정은 발견되지 않았다. 새 HEAD CPU CI의
실행 인수는 별도로 확인한다. late Ready fixture는 기존 runtime을 직접 다시 pump하며
공개 finish handle에 새 자동 회수 경로를 추가한 증거가 아니다. 이 fixture는 실제
ONNX/ORT 모델 로딩·GPU fence·새 native replay wrapper 실행 증거가 아니다.
factory/source/binary/provider 등록, 원래 전체 시계와 명시적 finish 결과, strict
Query/full recheck·whole cost·physical closure·utility 인수는 계속 남는다.

### 원래 전체 시계를 보존하는 CPU Fresh loader

`NativeInvocationBudget`은 caller의 원래 시작 S, 실행 마감 E, 전체 마감 W,
cleanup reserve D를 private checked fields로 보존한다. S<E<W, D>0,
E+D=W와 기존 유한 drain 조건을 검증하며 미래 시작 시각을 거절한다.
`load_pinned_cpu_fresh_for_invocation`은 기존 loader와 별도로 명시 선택하는
CPU Fresh 접점이다. options의 drain이 D와 같고 CPU provider·host 입력·1~2 intra
threads·record pages 없음·단일 CPU runtime pin인 경우만 실제 로딩을 진행한다.
정확한 public-memory cache 옵션은 독립 asset profile의 선언과 함께 고정하며
private Warm·resident/device 메모리와 혼동하지 않는다.

runtime load·backend load·owner construction의 각 동기 단계 전후에 원래 E와
cancel을 확인한다. 원래 오류, 실패 단계, S 기준 elapsed, E/W 초과와 cancel을
`NativeLoadFailure`에 보존한다. 직접 runtime/backend 실패에는 원래 구조화된
`BackendError`도 별도 소유하여 문자열 투영으로 원인·단계·bounded diagnostic을
잃지 않게 한다. 기존 owner constructor의 `RoleError`만 반환되는 경로에서는
해당 구조 진단 미관측을 그대로 남기며 전체 오류 journal로 표현하지 않는다.
실제 `WorkerOwner` 생성 직후 finish handle을
보존하여 이후 fallible adapter/runtime 구성 실패에서도 caller가 같은 W로
명시적 종료를 관측할 수 있게 한다. owner 생성 전 `finish=None`은 cleanup
미관측이며 실행 0이나 native resource release 성공을 뜻하지 않는다.

동기 ORT/session 호출은 이 API가 선점하지 않는다. 단계 전후 검사와 늦은 반환
거절은 hard timeout 증거가 아니며 외부 caller의 원래 시계·process supervision이
필요하다. cleanup에 새 now+D 창을 만들지 않는다. CPU config만으로 CUDA bundle
preload를 막을 수 없으므로 runtime load 전 실제 pin의 bundle 부재를 확인한다.
그 부재는 CPU provider 실행 성공을 인증하지 않는다. ORT의 process-lifetime
runtime pin과 실제 worker/session join·buffer release는 별도로 기록한다.

이 접점과 CPU Fresh replay wrapper는 별도 소스·검사 단위로 통합한다. loader의
pure/controlled fixture, 실제 wrapper compile, 실제 모델 replay와 원래 전체 시계
인수는 각각 구분하며 기존 Query/2·CPU action/1·replay input 등록 wire를 바꾸거나
실행·whole-cost·physical·utility 권한을 자동으로 부여하지 않는다.

### 실제 CPU Fresh 모델을 연결하는 frozen replay wrapper

`strategic_action::native_replay`는 `onnx-cpu`의 명시적 연결부다. 기존
`check_replay_inputs`로 원래 시작·실행 마감·전체 마감과 독립 등록을 검사한 뒤,
독립 asset profile과 원본 profile bytes·export manifest·runtime library를 대조한다.
기존 CPU action CLI와 semantic binary의 등록은 그대로 보존한다. 새 wrapper의
factory·source·replay binary 등록은 그 기존 실행 증거를 대신하지 않는다.

프로필의 `encoding_semantic_sha256`은 실제 Fresh 모델 입력의 기존 식별값이다.
이는 `SHA256(PALS_ENCODING_SCHEMA bytes || Rules digest)`이며, manifest의
Rules 전용 식별값과 구분한다. 공개 `pals_fresh_encoding_semantic_digest()`를
Fresh constructor·CUDA record 선언·wrapper가 공유한다. PrivateWarm capability의
인코딩과 Rules manifest 검사 의미는 유지한다. Rules digest만으로 profile context를
다시 봉인해도 입장을 거절하며, 로딩 후 실제 모델의 식별값도 정확히 대조한다.

선택한 경로는 invocation loader → 실제 `NativeRoleModel` →
`FreshReplayOwner`다. ReplyOnly는 기존 `run`의 2N 경로를, RepairEndpoint는
`run_with_repair`의 3N 경로를 사용한다. 원래 cancel·limits·generation을 전달하며
mock 결과를 native 결과로 대체하지 않는다. owner 구성 전 실제 finish handle을
보존하고, run 실패에서도 남은 store·attempt·report와 실제 관측 row를 수집한다.
runtime/owner Drop 뒤의 같은 W finish는 새 retry 창이나 runtime pump를 만들지
않는다. 완료 미관측 lease와 cleanup 실패는 계속 미관측·실패로 남긴다.

public-memory cache 선택은 독립 profile과 정확히 같아야 한다. CUDA bundle,
device public memory, private Warm 및 다른 provider는 native load 전에 거절한다.
CPU config나 pin 검증 성공만으로 실제 provider 실행을 인증하지 않는다.

입력·관측·출력의 예약은 실제 실행 전에 유한한 상한으로 검사한다. 기본 L4/P1의
ReplyOnly·RepairEndpoint는 기존 4MiB 출력 한도 안에서 각각 입장 가능하고, 더 큰
구성은 예약 부족으로 명확히 거절할 수 있다. 제한을 자동 축소하거나 mode를 바꾸지
않는다. compact JSONL의 외부 string escaping과 자유 Debug snapshot의 escaping을
구분하여 예약하며, row 실패 전에 이미 소유한 row는 보존한다.

오류는 원래 primary와 cleanup·observer·output secondary를 별도로 소유한다.
`NativeReplayError::serialized()`도 같은 원래 W를 적용한다. 입력 오류에 유효한
whole wall이 이미 입장됐으면 최초 시작에서 그 W를 복원하고, whole clock이 없는
malformed 입력은 미관측으로 구분한다. 직렬화 실패나 마감 초과가 원래 원인,
nonzero CPU work·unknown counter·앞선 row를 지우지 않는다. 원래 structured
backend error를 소유하는 것과 JSON의 bounded 진단 투영은 별도 범위다.

관측 출력은 실제 TaskRecord·attempt·report·Reply/Repair 및 native binding을
구분한다. formatter의 `complete`는 그 bounded snapshot의 완전성을 뜻하며 전체
typed/raw archive를 인증하지 않는다. raw float는 IEEE bits로 보존하고, publication,
Ready, 정상 소비·accepted context는 각각 기록한다. bytes 준비 시각은 caller의
delivery·flush·exit 시각을 포함하지 않는다.

이번 연결부에는 profile/source 입장, 두 mode의 예약, 원래 시계, 오류·관측 보존을
다루는 pure/controlled fixture 14개를 추가했다. 소스 검토와 실제 fixture 실행은
구분하고 새 HEAD CPU CI 결과로 실행 여부를 확인한다. 이 fixture는 실제 ORT 모델
로딩이나 registered native replay 성공을 대체하지 않는다. 별도 CLI 연결, 실제 CPU
모델 실행, strict Query/full recheck·whole cost·physical closure·utility 인수는
후속 검사로 남는다. GPU 검증은 사용자 지시에 따라 보류하며 실제 학습은 범위 밖이다.

### 별도 CPU frozen replay CLI의 등록·출력·오류 수명

`pals_frozen_replay`는 기존 `pals_cpu_task` CLI와 구분한 opt-in 예제다.
native 구현은 `onnx-cpu` feature 안에 두고, feature 없는 실행은 std-only 경로에서
명시적으로 거절한다. 기본 빌드에 선택 serde/ORT 의존성을 자동 활성화하지 않는다.
기존 CPU action CLI의 wire와 실행 증거는 그대로 보존한다.

인자는 `--expected-pins ABS`, `--expected-sha256 HEX`, `--expected-bytes POS`,
`--asset-profile ABS` 네 쌍으로 한정한다. frozen request는 2MiB 이하 원문 stdin이다.
expected 원문은 인자로 받은 독립 bytes/SHA에 대조하고, closed expected·launch
schema로 request·launch·profile·모델·CPU runtime 등록을 연결한다. model·provider
선택의 기본 경로·환경 변수 fallback이나 stdin에서 expected를 유도하는 경로는 없다.
RuntimeCache의 실제 CPU library bytes/SHA와 bundle 부재를 독립 profile에 대조하며,
cache 검증을 실제 ORT provider 로딩 성공으로 해석하지 않는다.

새 실행 image는 독립 replay binary pin에 대조한다. Linux는 열린 `/proc/self/exe`
inode, 다른 OS는 `current_exe` 경로 해시 범위로 구분한다. replay SHA는 역사 CPU
및 semantic binary SHA와 각각 달라야 한다. legacy bytes만 다르게 선언해 같은 SHA를
등록하는 경우도 wrapper에서 거절한다. 이 분리 검사는 과거 CPU·semantic 실행이나
provider loaded image의 provenance를 새로 관측했다는 권한을 주지 않는다.

최초 S는 인자·stdin·독립 파일·image hash·runtime cache·dispatch·출력까지 포함한다.
성공 입력의 W/E/D/output 선언은 독립 transport와 정확히 같아야 한다. 입력 오류가
더 짧은 유효 request W/E나 더 작은 output을 이미 알고 있으면 오류 처리에도 기존
한도와의 최소값을 적용하고 두 선언·불일치를 별도로 보존한다. 원래 typed input
cause는 유지하며, 오류 직렬화에도 유효 마감·출력 한도를 적용한다. malformed clock은
미관측으로 구분하고 새 상대 시간창을 만들지 않는다.

정상 `CheckedReplayInputs`를 받은 뒤 transport와 W/E/D/output이 달라 거절하는
경로도 같은 원칙을 따른다. 실제 checked getter에서 얻은 마감·출력과 원래 resource
선언을 먼저 보존하고 양쪽 한도의 최소값을 적용한 뒤 불일치를 거절한다. transport와
request의 원래 cleanup 선언을 혼합해 새 cleanup 값을 추정하지 않는다. `admitted_input_clock`
관측은 입력 admission의 한도 대조이며 Query·provider·실행·물리 완료 권한이 아니다.

native 성공·실패 JSON bytes는 재봉인하거나 다시 직렬화하지 않고 outer envelope에
그대로 넣는다. CLI header·newline까지 원래 출력 상한에 포함하고, 부족하면 원래
body와 오류를 소유한 채 전달을 거절한다. 준비 시각·확인된 write prefix·flush·exit·
물리 종료를 서로 구분한다. header는 출력 소비·process exit·physical closure 성공을
미리 선언하지 않는다.

stdout/stderr는 전체 가능한 frame bytes를 공유 credit에 선예약한다. timeout이
writer 종료나 zero-byte 성공을 뜻하지 않으며, 실제 OS writer가 종료하기 전에는
미완료 예약을 반환하지 않는다. 확정된 미사용량만 돌려준다. 원문과 typed native/input
error evidence를 같은 owned bundle로 pending writer에 보존하고, 원래 write/flush
결과·io cause는 알림 채널과 별개의 유한 terminal record에 남긴다. 늦은 오류·부분
출력·진단 출력의 secondary 실패를 원래 search/backend 오류와 합치지 않는다.

기존 CPU matrix에는 feature 없는 예제의 compile/test 한 단계만 추가한다. 새 CLI의
순수·제어된 I/O fixture와 실제 실행은 별도 인수이며, default compile 성공은 production
CLI 실행 성공을 뜻하지 않는다. 동기 파일/ORT/pipe의 강제 중단, native 진단을 포함한
전체 process output 상한, 같은 W의 external launch/reap·loaded provider·physical
증거는 외부 supervisor가 따로 수거해야 한다. pending Arc 보존은 외부 내구성 저장의
증거가 아니다. 독립 등록과 원문을 준비하는 caller 연결, 실제 CPU 모델 replay 및
strict Query/full recheck·whole cost·physical·utility는 계속 후속 인수로 남긴다.

### Semantic receipt 생산 범위의 독립 등록

기존 semantic library는 `dispatcher_compared_verified_argument` 범위의 flat receipt를
만든다. 기존 `pals_cpu_task --prepare-semantic` CLI의 성공 출력은 같은 flat receipt에
Linux의 `linux_loaded_executable_inode` 또는 다른 OS의 `current_exe_path_hash` 범위를
기록해 직렬화한다. 이 CLI 출력 안에 원래 library receipt가 따로 중첩되지는 않는다.
frozen replay 연결을 위해 receipt의 scope를 다시 쓰거나 원문을 재직렬화하지 않는다.

기존 `check_replay_inputs`·`dispatch_started`와 CLI expected `/1`은 library 범위만
받는 의미를 유지한다. explicit semantic-scope API와 별도 CLI expected `/2`는 독립
등록한 `expected_semantic_receipt_producer_scope`를 필수 closed enum으로 받아, pin과
context를 검증한 원문 receipt의 범위와 정확히 비교한다. 원문·현재 OS·현재 replay
image에서 기대 범위를 선택하거나 세 범위를 무조건 허용하는 방식으로 바꾸지 않는다.
`ReplayInputRequest`·`ReplayExpectedPins`의 기존 wire와 context는 유지한다.

explicit 선택은 성공의 input audit와 실패 기록에 기대 선언으로 보존한다. 기존 경로의
`None`은 JSON에서 생략하며, 기존 `binary_pin_scope`는 replay image의 독립 caller
선언 범위를 계속 뜻한다. 기대값을 전달했다는 기록이 과거 producer image의 재관측,
scope 일치 성공, strict Query/2 전체 검증이나 물리 종료 권한을 만들지는 않는다.

CLI는 독립 expected 원문에서 알려진 원래 W·cleanup·output cap을 먼저 적용한 뒤
version·scope를 검사한다. `/2`의 missing·null·unknown·duplicate·잘못된 타입과 `/1`의
새 scope 필드는 거절한다. 이 거절에 새 시간창이나 더 큰 출력 cap을 주지 않는다.
유효 closed `/2`를 읽은 뒤 E가 만료되는 경로에서도 기대 선언을 먼저 보존하고
마감을 거절한다. 파싱에 실패한 scope를 관측값으로 채우지는 않는다.
원문 pin·context 변조, 범위 불일치, legacy 생략, 실패의 typed 원인·원래 한도와
negative authority는 CPU fixture의 별도 검사 대상이다. 이 연결의 소스·fixture 검사는
등록된 실제 CPU 모델 replay·외부 supervisor 종료·utility 수집의 실행 인수를 대체하지 않는다.

### Strict caller 원문과 Rust replay wire의 준비 접점

`strategic_replay_caller`는 기존 Query/2와 `PreparedStrategicCpuAction`을 재검증하고,
caller가 보존한 semantic request·whole receipt·source·registration·before result·launch
observation·common query의 일곱 원본 bytes를 기존 semantic factory에 다시 입장시킨다.
선택된 action과 semantic slot, parent/current/frozen 및 실제 encoding이 같은지 확인한다.
원문을 잃었으면 거절하며, receipt dictionary를 재직렬화해 과거 pin을 복구하지 않는다.
생산 범위는 독립 등록 원문과 외부 기대값을 대조한다. receipt 결과나 현재 OS를 보고
기대 범위를 선택하지 않는다. 현재 strict semantic factory가 지원하는 두 image 범위를
넘겨 허용하지 않으며, 이 준비 결과가 실제 과거 image를 다시 관측했다는 뜻은 아니다.

새 Python 접점의 원문 합계 12MiB·pin metadata 64KiB·추가 준비 scratch 8MiB는
해당 접점의 유한 정책 credit이다. 기존 Query/action의 원문 한도는 계속 적용한다.
이미 존재하는 capability의 resident 메모리나 Python/Rust allocator의 물리 peak를
측정하거나 완전히 제한한 수치가 아니다. 각 재검증은 같은 유한 caller deadline을
사용하며 새 시간창을 만들지 않는다. Python은 Rust의 2N/3N·role/store 요구량을
재계산하거나 예산을 자동 확장하지 않는다.

Rust의 `prepare_replay_request`는 세 원문을 borrowed bytes로 받아 UTF-8·독립 pin을
대조하고, 기존 replay 입력의 mode/config/resources와 expected 선언을 사용한다.
이 세 원문은 새 독립 `ReplayConsumerRegistration`, prepared action/1, whole semantic
receipt다. Python 접점의 과거 semantic registration은 별도 역사 근거이며 새 replay
consumer registration 자리를 대신하지 않는다. Python handoff는 action과 receipt 및
역사·선택 identity의 부분 연결만 제공한다. 새 consumer registration·replay binary·
source·factory·assets와 완전한 `ReplayExpectedPins`는 독립 caller가 별도로 공급한다.
새 바깥 JSON만 직렬화하며 세 embedded 원문의 공백·줄바꿈·키 순서와 bytes는 보존한다.
`ReplayInputRequest`·`ReplayExpectedPins`·기존 CLI expected `/1`과 `/2`의 wire 및
context 의미는 유지한다. 완성한 wire는 같은 S·scope·expected로 기존 scoped checker에
통과시켜 실제 Rules와 원래 remaining, ReplyOnly2n/RepairEndpoint3n의 요구량을 검사한다.

준비 예산은 outer wire 최대 2MiB, construction credit 최대 16MiB다. escaping을 포함한
길이를 할당 전에 count하고 `3 × Sraw + 2 × C + Q + 64KiB`를 checked arithmetic으로
입장시킨다. Sraw는 세 원문 길이 합계, C는 기존 closed context의 직렬화 길이,
Q는 64hex context가 포함된 최종 wire 길이다. credit은 context 계산의 원문 문자열
복사 세 벌·canonical backing·최종 wire 및 고정 topology allowance에 대한 보수적 정책이며
실제 allocator peak의 측정값이 아니다. 최종 wire의 backing과 쓰기는 cap 및 원래
E/W 안에서 확인한다. 뒤따르는 validator의 typed/Value/Rules heap은 이 credit의
완전한 메모리 상한으로 주장하지 않으며 외부 memory supervision과 별도로 인수한다.
기존 context 함수는 body Value, `json!` array Value, canonical sorted Value를 만든다.
현재 사용 중인 [serde_json 1.0.145의 macro](https://github.com/serde-rs/json/blob/v1.0.145/src/macros.rs)
및 [Value serializer](https://github.com/serde-rs/json/blob/v1.0.145/src/value/ser.rs)의
소유 문자열 복사를 대조하여 이 세 payload를 예약한다. hash 함수와 의미는 유지한다.

성공 타입은 private immutable wire·실제 artifact pin·준비 audit와 원래 마감만 제공한다.
거절된 최종 후보와 typed 원인은 오류의 소유로 보존하며 성공 타입으로 반환하지 않는다.
입장 가능한 원래 자원 선언은 이른 UTF-8·pin·준비 budget 거절에도 S/W/E와 진단 cap을
남긴다. 모델/provider/dispatch, durable caller 등록, whole cost·물리 종료·full Repair
recheck·utility·학습 target의 권한은 발급하지 않는다. 실제 CLI expected/launch 파일의
결과 전 등록과 CPU 모델 실행·종료 자료 회수는 이 순수 준비 단위 이후의 별도 인수다.

### Rust의 결과 전 frozen replay 파일 준비

`replay_launch_preparation::prepare_replay_launch_bundle`은 기존 ReplyOnly2n 및
RepairEndpoint3n 요청을 파일로 준비한다. 새 replay registration·prepared action·whole
semantic receipt·CPU Fresh profile의 독립 원문과 expected 선언을 받고, 기존 생성자와
scoped Rules/resource checker를 사용한다. native wrapper의 동일한 role/check 수와
공개 output 예약 산식으로 원래 출력 한도를 확인한다. profile도 기존 closed decoder를
공유한다. 이 순수 접점은 실제 runtime·export 파일을 읽거나 모델/provider를 만들지 않는다.

CLI의 `ExpectedPinsWire`, expected `/2`, launch assets DTO는 이 모듈과 공유한다. 기존
V1 outer와 V1/V2 parser·실행 순서는 유지한다. 현재 실제 CLI는 binary domain 선언을
먼저 검사하고 stdin·Rules/resource·독립 assets/profile을 확인한 뒤 실행 image를
관측한다. 입력 전 domain 선언 검사를 실제 loaded image 관측으로 표현하지 않는다.
기존 private flat wire의 field/type/closed serde와 context revision은 바꾸지 않는다.

네 원문 파일과 새 replay input·launch assets·expected의 일곱 payload를 caller가
소유한 새 디렉터리에 저장하고, manifest를 마지막에 게시한다. caller root는 기존의
절대 UTF-8 경로이며 한 component의 새 child만 만든다. payload는 `create_new`,
확인된 write return 수, file sync 및 동일 handle의 고정 4KiB chunk readback으로
검사한다. pending manifest의 확인이 끝난 뒤 정상 이름으로 옮기고 directory sync를
시도한다. 실패는 원래 typed 원인과 실제 작성/readback 범위, 준비된 요청·payload·부분
직렬화 bytes를 소유해 반환한다. 기존 디렉터리·파일을 덮어쓰거나 자동 삭제하지 않는다.

최종 게시 후 실패하면 원래 W 안에서 정상 manifest의 격리를 시도한다. 격리 실패와
정상 이름의 부재 미확인을 secondary로 보존하며 성공으로 바꾸지 않는다. 파일 존재만으로
실행 권한을 주지 않는다. 성공 객체도 다른 프로세스의 source/build/binary/factory 등록,
실제 모델 실행·종료·전체 비용·utility를 인수하지 않는다. 후속 launch는 독립 등록·pin과
같은 원래 caller 마감을 다시 확인해야 한다.

추가 publication IO 및 backing 예산은 각각 최대 16MiB다. bounded metadata bootstrap
128KiB를 먼저 확인한 뒤 assets와 expected를 backing 없는 count/hash writer로 측정하고
합계를 입장시킨다. IO 정책은 `2 × (네 원문 + 생성 요청 + assets + expected + manifest
최대 64KiB) + 8bytes`이며 마지막 EOF 확인을 포함한다. backing 정책은 네 원문 복사,
assets·expected·manifest backing과 metadata allowance를 포함한다. 기존 생성 요청의
소유권은 이동하며 construction credit과 별도로 표시한다. policy credit은 기존 typed
validator·canonical JSON·allocator의 전체 peak나 RSS 측정값이 아니다.

모든 시간 검사는 같은 process-local S/E/W와 cancel을 사용하는 cooperative checkpoint다.
동기 OS I/O의 강제 timeout, hostile concurrent parent 교체 방어, Instant의 프로세스 간
전송을 증명하지 않는다. file sync/readback과 close 결과·crash persistence는 구분한다.
Windows directory metadata persistence는 unknown으로 남기고 Unix directory sync 관측도
별도 getter로 제공한다. 준비 manifest에는 spawn·native result·physical·whole cost·utility·
training 권한을 부여하지 않는다.

신규 CPU fixture의 소스는 13개이며 Unix link fixture 하나는 Windows에서 미실행이다.
실제 Rules 준비를 사용한 두 mode의 원문/manifest 정합, 원래 자원 부족·마감·cancel·
profile 원인, publication credit, 기존 디렉터리 충돌, 경로·링크 거절, bounded readback과
격리 실패, escaping/부분 직렬화, 공유 codec와 독립 scope 불일치를 다룬다. controlled
tempfs 및 순수 준비 검사이며 actual CPU task·모델/provider·child process를 실행하지 않는다.
이 소스 단위의 CPU CI와 등록된 실제 replay·external supervisor·strict Query 환류·별도
4N full opponent recheck·whole cost·물리 종료·utility 인수는 각각 구분한다.

## 2026-10-09 — 명시적으로 선택하는 4N 실제 C 수순 재검사

`FreshReplayOwner::run_with_opponent_recheck`는 기존 2N Reply와 3N Repair의 다음
단계를 search-local 접점으로 제공한다. 두 기존 mode는 이 경로를 자동 선택하지 않는다.
별도 scope는 `rz-pals-frozen-parent-repair-opponent-continuation-replay/1`이다. 추가
타입은 `replay/opponent_recheck.rs`에 두며, 같은 Rules·역할 출력 인수·CPU task·원래
마감과 단일 finish hook을 사용한다. 이 소스 단위 당시 CLI·등록·launch 명세는
2N/3N 전용이었다. 다음 절은 별도로 추가한 명시적 4N 연결을 설명한다.

실제 완료한 세 번째 unrestricted Repair 관측과 게시된 Repair line/revision을 먼저
확인한다. 원래 공격 이후 달라진 자기 수 다음의 첫 상대 차례를 찾고, C가 그 상태에서
기존 응답과 다른 합법 응답을 선택하게 한다. 이후 horizon까지의 수들도 실제 C policy
호출로 생성한다. Repair suffix를 복사해 이어 붙이지 않는다. 각 호출의 확인된 Rules
state·논리 context·출력 인수 여부·선택한 수를 보존하고, 부분 실패에도 앞선 관측과
생성된 수순을 남긴다. 모델 정책이 제시한 대안은 검증 대상이며 최선 방어의 증명이 아니다.

`repair_opponent_replay_requirements(L,P,N)`는 `L-P >= 3`, 고정 CPU 상한 4N,
기존 3N 역할 상한에 `L-P-2`회의 C 호출, record 5개와 stage 4개를 요구한다.
노드·관측·line chunk의 추가 예약도 checked 산식으로 선언한다. 생성자 소유 저장소
한도와 원래 자원·마감이 부족하면 실행 전에 거부한다. 이 선언은 peak 측정값이 아니다.
새 CPU 검사는 같은 H1·profile/config를 가진 새 checker/TT에서 unrestricted
`AnalyzePosition`, 빈 root moves, input revision 0, Start로 수행한다. 새 stage/task의
관측을 보존하고 세 번째 Repair endpoint의 근거를 덮어쓰지 않는다. Rules terminal은
별도 종료 결과이며 네 번째 CPU 관측을 만들어 넣지 않는다.

신규 소스 검사 13개는 C의 응답과 다음 수가 모두 Repair와 달라지는 예, fresh 네 번째
실행·정확한 history/context·기존 3N 선택 보존, anchor 부재, 자원 부족, 부분 CPU,
마감·취소·모델 오류·잘못된 출력·물리 완료 불명을 다룬다. 이 문단은 검사 범위의
설명이며 실행 성공은 정확한 SHA의 CPU CI 자료로 별도 확인한다. native RequestId·
모델/provider 실행·strict Query 환류·external supervisor·전체 비용·물리 종료·utility·
학습 target 인수는 이 소스 경로만으로 성립하지 않는다. 실제 학습은 범위 밖이며
GPU 검증은 사용자 지시에 따라 보류한다.

새 L=5 합성 fixture의 초기 quiescence 한도 4에서는 세 번째 Repair 관측이 부분
완료여서 C 단계에 진입하지 않았다. 해당 입력·조건을 음성 검사로 보존하고
`QuiescenceLimit`·task failure·새 C 및 네 번째 실행 부재를 확인한다. 양성 경로의
합성 fixture는 생성자에서 quiescence 16을 명시하고 네 단계에 동일하게 적용한다.
제품·등록 실행·기존 2N/3N fixture의 설정이나 완료 기준을 변경하거나 부분 완료 후
자동으로 한도·마감·자원을 바꾸지 않는다. 이전 CI의 실패 원문도 별도로 보존한다.

## 2026-10-09 — 4N 입력 등록·native dispatch·durable launch 연결

`repair_opponent_4n`은 입력 `rz-pals-frozen-replay-inputs/2`, 등록
`rz-pals-frozen-replay-consumer-registration/2`를 명시적으로 사용한다. 기존
`reply_only_2n`·`repair_endpoint_3n`은 각각 기존 `/1` schema, context domain과
직렬화 형식을 유지한다. 새 `opponent_recheck_source` pin은 `/2`에서 필수이고
독립 기대값·원문 registration context·컴파일된 실제 소스 바이트와 각각 대조한다.
`/1`에는 이 필드를 넣지 않는다. 필드 부재는 기존 바이트를 유지하지만 명시적
`null`, 중복, 미지원 필드·schema 교차·크기·digest 불일치는 거부한다. 이 source
비교와 과거 binary 등록, 현재 OS 실행 이미지 관측은 별도 근거다.

`replay_mode_requirements`가 입력 인수·launch 준비·native 실행의 공통 자원 산식을
제공한다. 2N의 기존 보수적인 Repair role/store 예약은 유지하며 실제 observer 호출
상한 L+1과 CPU stage 2개를 구분한다. 3N과 4N은 각각 실제 role 상한·stage 3/4개를
사용한다. 새 4N 입력은 원래 prepared action의 remaining nodes/wall/output에 들어가야
한다. 원래 N·H1·CPU profile·시계·정리 reserve는 바꾸지 않는다. constructor나
publication 함수가 부족한 이전 allowance를 늘리지 않는다.

native dispatcher는 동일 `FreshReplayOwner`의 실제 `run_with_opponent_recheck`를
호출한다. 기존 역할 observer가 준비·물리 완료·논리 출력 인수를 계속 기록한다.
관측 출력 `/2`에는 새 `opponent_state`를 별도로 두어 실제 anchor, 생성된 수순,
수별 logical context·accepted·selected와 endpoint를 보존한다. 네 번째 stage의
실제 CPU task/report/attempt/observation은 기존 세 번째 Repair stage와 분리한다.
2N/3N 출력 `/1`에는 새 필드를 직렬화하지 않는다. 실패에도 이미 생긴 stage·수순·
scalar work를 포착하고 기존 typed 원인·cleanup·observer 실패를 보존한다. CPU ID와
native RequestId를 같은 것으로 취급하지 않으며 완료 관측을 utility 권한으로 승격하지 않는다.

출력 상한 4MiB는 그대로다. 4N은 별도의 관측 layout으로 큰 prepared/terminal 행을
각각 32KiB, accepted context를 8KiB, 작은 행을 1KiB, 역할당 최대 12행으로 제한한다.
역할별 backing은 82KiB, 새 상대 상태 snapshot은 32KiB이며 JSON escaping·전체
header·stage·outcome 예약을 모델 로딩 전에 검사한다. L=5/P=1의 최대 14 role
호출은 기존 상한에 들어간다. 더 긴 입력이나 관측 행이 예약을 넘으면 명확히 거부·
실패하고 기존 결과를 보존한다. 출력 상한 확장·원시값 생략 후 성공 처리·자동 재시도는
하지 않는다. 이 한도는 직렬화 예약이며 RSS·allocator peak가 아니다.

durable launch 준비는 새 입력과 등록 원문을 그대로 보관하고 동일 공통 산식으로
native 출력 한도를 검사한다. 기존 expected transport `/2`는 공통 artifacts DTO를
통해 새 pin을 운반한다. transport schema와 입력/registration schema는 다른 domain이다.
manifest 준비만으로 모델 로딩·child spawn·기존 Query/2 재검증·물리 종료가 성립하지 않는다.

이 단위는 입력 검사 6개, native source/예약/실패/실제 CPU dispatch 포착 검사 4개,
durable launch 검사 2개를 추가한다. 실제 CPU 포착 fixture는 합성 RoleModel·고정
quiescence 16과 실제 네 단계 checker를 사용한다. ORT/GPU/provider/외부 child는 실행하지
않는다. 기존 CLI 소비 검사와 2N/3N 검사도 정확한 SHA의 CI에서 별도로 확인한다.
실제 등록된 신경망의 4N 실행·external supervisor·strict Query 결과 환류·전체 비용·
utility/target 인수는 아직 미실행이다. 실제 학습은 제외하고 GPU 검증은 계속 보류한다.

## 2026-10-09 — 등록된 4N 결과의 Rust 관측 소비 경계

`native_replay::query_prior`의 구현은 같은 디렉터리의 `replay_prior.rs`에 둔다. native
로딩·실행·관찰자와 분리되는 책임은 결과를 다음 private V 입력으로 보존할 수 있는지
판정하는 것이다. Rules·CPU checker·native lease를 다시 구현하지 않는다.

명시적 `dispatch_started_observed_with_semantic_scope`는 같은 원래 S/E/W 안에서 실제
dispatch와 최종 출력 검사를 수행하고, 하나의 결과 객체로 반환 bytes와 읽기 전용 typed
관측을 함께 넘긴다. 4N은 새 native 관측 `/3`에 `rz-pals-frozen-replay-query-prior/1`
projection을 추가한다. 기존 byte 진입점·CLI·durable launch는 2N/3N의 `/1`과 4N의
`/2`를 유지한다. 입력·registration `/1`·`/2`, 기존 Python Query/2의 wire·digest와
공통 계약 revision은 바꾸지 않는다. 기존 CLI에 새 결과를 자동 활성화하지 않는다.

projection은 네 CPU 단계의 정확한 완료·요청/완료 깊이·완료 이유·PV move16·실제
nodes/qnodes/TT hits, 원래 H1 조건의 metadata hash, Repair와 실제 새 상대 수순을
포착한다. 단계 4개·PV 96수·수순 16수의 고정 inline 저장소를 사용하며 동적 PV 복제나
Debug 재해석은 없다. 실제 terminal은 세 CPU 단계만 직렬화한다. 범위를 넘으면
실패로 보존하며 잘라낸 성공이나 추가 예산을 만들지 않는다. projection JSON의 상한
16KiB는 기존 512KiB header 예약 안에 두고 동일한 전체 출력 상한을 유지한다.

`consume_native_replay_prior`는 등록·원래 action/semantic receipt·parent/current/frozen/
encoding·Query/catalogue/before/prior·factory/source의 독립 expected 값과 입력 audit를
대조한다. 같은 실제 stage의 scalar와 projection도 비교한다. 미완료·frontier·재사용
깊이·누락 비용·취소·오류·native 완료 불명·버퍼 미해제는 완료 입력으로 승격하지 않는다.
CPU 점수·실행 ID·식별 hash는 신경망 feature로 사용하지 않는다. compiled projection
source hash는 metadata로 기록하며 이를 독립 expected source나 전체 build 인수로
대체하지 않는다.

E 검사는 실제 `run_owner` 반환 시각에서, W 검사는 정리·관측·출력 뒤에서 수행한다.
기존 `execution_deadline_exceeded`가 정리 종료 시점까지 포함한다는 사실을 보존하며,
정리가 E를 지나 W 안에서 끝난 경우를 작업 지연으로 재라벨하지 않는다. 반환 뒤
취소·최종 출력 실패는 prior readiness도 실패 상태로 철회하고 이미 관측한 작업은 남긴다.

완료 CPU 범위와 native 종료를 모두 확인해도 상태는 `pending_caller_chronology`다.
새 소비자는 strict dataset/current selector·frozen producer 재입장, 외부 child의
spawn/loaded image/EOF/reap/process group·원래 시작 시각, 다음 before/prior ledger를
검증하지 않는다. 새 결과를 기존 Query/2 completed prior로 전달하지 않으며 utility
group은 0, utility/target/training 및 전체 causal bridge 권한은 false로 유지한다.
실제 등록된 ONNX 4N 실행과 그 결과의 strict Query 환류·전체 비용·utility 인수는
후속 단계다. CPU source 검사를 실제 모델·GPU 실행으로 보고하지 않는다.

## 2026-10-09 — 관측 결과 선택을 실제 frozen replay CLI에 연결

`prepare_observed_replay_launch_bundle`는 등록된 `RepairOpponent4n`에 한해서 새
`rz-pals-frozen-replay-cli-expected/3`를 게시한다. `native_result=query_prior_v1`은
명시적 결과 선택이며 입력·출력에서 추론하지 않는다. 새 transport는 완전한 기존 `/2`
기대 명세를 `expectations`에 소유권 이동해 담고, 원래 wall·cleanup·output을 바깥에도
기록한다. CLI는 안팎의 값이 모두 일치하고 내부 schema가 `/2`인 경우만 인수한다.
중복·누락·미지원 값·추가 필드는 거부한다. 기존 예상 bytes의 재직렬화를 새로운
독립 기대 pin이나 strict Query 의미 검증으로 대신하지 않는다.

기존 `prepare_replay_launch_bundle`은 기존 `/2` 기대 bytes와 준비 manifest `/1`을
유지한다. 새 API만 준비 manifest `/2`의 `requested_native_result`를 게시하고, 그
schema를 별도 context digest 도메인으로 쓴다. 동일한 원본·입력·asset 경로·S/E/W를
사용하며 기존 직렬화·게시·readback credit과 native output 상한을 늘리지 않는다.
이미 실행한 모델 결과를 준비 manifest에 미리 채우지 않는다. 준비가 관측하는 것은
파일 게시이며 `spawn_observed`·`native_result_observed`와 모든 권한은 false다.

실제 `pals_frozen_replay` CLI는 기대 `/3`의 선택과 인수한 4N mode를 대조한 뒤,
asset/cache 로딩 전 `ObservedPrior` dispatch 경로를 고정한다. 이 경로는 기존 같은
시계·같은 assets로 `dispatch_started_observed_with_semantic_scope`를 직접 호출하고,
완료 결과 bytes를 기존 bounded delivery 경로에 넘긴다. 이미 직렬화한 결과를 읽기
전용으로 넘기며 동적 PV나 native 관측을 추가로 복제하지 않는다. `/1`·`/2`는 기존
byte dispatch를 계속 사용한다. 기본 feature의 native 거부·기존 argv·입력과 registration
schema·공통 계약은 바꾸지 않는다.

선택한 결과 종류는 CLI 준비 header와 오류 진단에 남긴다. 미선택 경로에서는 새 필드를
생략한다. 이 metadata는 stdout 전달 완료·프로세스 종료·loaded provider·물리 종료의
자기 증명을 만들지 않는다. 반환 관측의 `pending_caller_chronology`와 utility group 0을
유지한다. 다음 caller는 원래 예산 아래의 외부 bounded capture·EOF·reap·group 정리와
loaded-image 관측, 독립 등록 자료 및 next before/prior chronology를 직접 연결해야 한다.
원래 process-local Instant를 다른 프로세스로 전송한 것처럼 표시하지 않는다.

검사에는 실제 durable 4N 게시의 원본·기대 bytes·readback·시계 보존과 2N/3N의 게시 전
거부를 포함한다. CLI의 명시 선택·기존 경로·안팎 clock·version·lane·duplicate/extra 필드
검사도 추가한다. 이 소스 단위의 CPU CI 결과는 정확한 HEAD에서 별도로 기록하며, 실제
등록된 ONNX 4N 실행·외부 supervisor·strict Query 환류·전체 비용·utility 인수는 남아 있다.
GPU 검증 보류와 실제 학습 제외를 유지한다.


## 원래 실행 창을 유지하는 Rust replay caller 후속 구현

이 단위는 기존 `535b52b`의 관측 4N 준비·CLI 연결을 arena의 실제 프로세스 감독 접점까지 연결한다. GPU 보류, 실제 학습 제외, 로컬 heavy CPU 미실행, PR #23 Draft를 유지한다. 정확한 실행 SHA의 CPU CI를 관측하기 전에는 아래 새 검사들을 통과로 기록하지 않는다.

- `OriginalProcessWindow`는 준비 소유자가 가진 process-local S/E/W를 전달한다. serialized duration이나 자식 시작 시각으로 새 실행 창을 만들지 않는다. 새 `supervise_input_in_original_window`는 1..2MiB frozen 입력을 기존 nonblocking pipe·pinned ELF·전용 process group·waitid/reap 경로로 전달한다. 기존 UCI protocol 입력의 64KiB·마지막 newline 제한과 상대 시간 API는 유지한다.
- 실행 중지 시각은 원래 E다. cleanup의 KILL과 최종 관측은 원래 W로 제한하고, 늦게 반환한 관측은 전달 완료를 통과하지 못한다. blocking OS preflight/spawn/파일 관측에 hard timeout을 보장하지 않는다. 자식의 내부 시계와 부모 S는 같은 process-local Instant가 아니므로, 부모의 절대 E/W 감독이 필수이며 이를 원본 시계의 process 간 직렬화 전송으로 주장하지 않는다.
- 새 전달 관측은 확인된 input write 수·stdin 종료·stdout/stderr EOF·S 이후 supervisor/spawn/반환 offset·최종 취소·소유권 상태를 기록한다. 정상 exit 0, 두 EOF, 전량 전달, group Gone, 실제 reap, pending child 없음, 오류 없음, W 이내 반환을 모두 요구한다. 기존 receipt의 elapsed는 기존 private supervisor 구간이며 S부터의 전체 offset과 구분한다. 관측은 역직렬화로 발급할 수 없다.
- `pals-collection-onnx`의 `pals_replay::supervise_prepared_observed_replay`는 prepared bundle을 이동하여 보존한다. 새 manifest /2·explicit expected /3·QueryPriorV1·등록된 4N만 받으며 old /1·/2를 자동 승격하지 않는다. 원래 whole/cleanup/output 선언과 단일 child 정책을 일치시킨다. 등록 replay binary를 caller의 열린 descriptor에서 bounded read_at으로 hash 검증하고, 원래 cwd inode와 준비 파일·manifest를 실행 전후 다시 검증한다. caller pin 검증은 native loaded image·model/source admission의 대체가 아니다.
- 요청 backing은 bundle에서 빌려 쓰며 전량 복제하지 않는다. binary 최대 512MiB, 검증 read credit 최대 544MiB를 별도 정책으로 제한하고, 모든 검증은 원래 E 또는 종료 후 W를 사용한다. descriptor 검증은 caller의 읽기 위치를 바꾸지 않는다. read credit·metadata/pin mismatch·취소·시간 초과는 거부한다. 경로/metadata 검사와 periodic process-group/tree 관측은 hostile concurrent mutation·escaped child·kernel cgroup/quota 집행을 증명하지 않는다.
- spawn 전 실패는 bundle을 소유한 typed failure를 반환한다. spawn 후 readback 실패는 capture 안에 기록하여 stdout/stderr·process receipt·pending child를 버리지 않는다. 사용자는 ownership-lost 또는 cleanup Unverified를 격리하고 후속 실행/결과 인수를 거부해야 한다. Drop은 durable 입력을 삭제하거나 미완료 child/모델 물리 완료를 인증하지 않는다.

새 CPU 검사는 원래 창의 순서·expiry·extension 거부, 실제 native cat의 96KiB 무개행 전달/EOF/reap, input을 읽지 않는 sleep의 E 중지, 출력 제한/취소 거부, 기존 protocol 입력 제한 유지, caller 정책 치환 거부, 실제 descriptor hash의 읽기 위치·credit·pin·원래 guard 검사를 포함한다. 실제 등록된 ONNX 4N bundle 실행·native 물리 closure·strict Query next-prior 인수·whole-action cost·utility/paired 성과는 이 검사와 구분하며 계속 미인수다. 전체 P0~P6의 잔여 구현과 CPU 실제 실행 조건도 유지한다.


### replay caller 최초 CI 실패와 종료 관측 보완

`f5e857a`의 CI `37864492893`은 네 job의 종료를 모두 확인했다. model CPU와 bindings는 성공했으나 Linux/Windows workspace는 `u64` process 출력 상한과 `usize` transport 선언의 E0308/E0277 비교 오류로 실패했다. 새 transport 검사 성공으로 기록하지 않으며 원시 로그를 별도 보고서 루트에 보존한다. 수정은 checked 정수 변환으로 선언값을 유지하며 변환 실패를 거부한다.

추가 검토로 원래 E 이전의 leader exit 관측을 W 이내 정리 완료와 분리했다. 첫 waitid 종료 관측의 S 이후 offset과 E 이전 여부를 기록하며, 해당 관측 없이 전달 완료를 발급하지 않는다. OS가 종료를 늦게 보여 주는 경우도 소급하여 통과시키지 않는다. 기존 실제 cat 검사에 E 관측 누락의 거부를 추가했다. 기존 receipt·protocol·준비/CLI wire·공통 계약은 유지하며, 수정 HEAD의 정확한 CPU CI 인수는 PR 설명과 외부 SHA·로그 영수증으로 별도 확인한다.


## 일반 UCI 탐색의 실제 C 후속 수순 — 별도 S lane

원래 계획의 Repair 이후 재공격을 제품 탐색에 연결한다. 기존
`same-repaired-line-once-v1`은 C의 새 응답 한 수 뒤에 기존 Repair의 나머지 수순을
Rules로 재생한다. 이 의미와 기존 생략/default lane·V1/V2 기록은 유지한다.
새 명시 옵션 `--pals-post-repair-recheck=actual-opponent-continuation-v1`은
원래 첫 착수를 보존하고, 대체 응답 이후 매 실제 counter-prefix에서 C Reply를 호출한다.
C는 상대와 자기 차례를 포함한 제한된 후보 수순을 생성하며, 기존 Repair suffix를
끼워 넣지 않는다. 새 수순의 길이는 수락된 Repair 길이를 넘지 않는다.

- 기존 `ranked`의 정확한 Rules 합법 수·context·출력 수락을 사용한다. 모델과 CPU
  namespace·value resolver는 바꾸지 않으며 새 역할 호출도 기존 전역 예산에 청구한다.
  같은 deadline/cancel/store 한도와 native 역할 평가기의 물리 수명을 재사용한다.
- 완전 길이의 counterline과 같은 완료 CPU depth/namespace 또는 두 Rules 종료
  근거가 있을 때만 해당 Repair line을 조건부로 반박한다. 부분/잘못된/늦은 출력,
  역할 한도 소진, 비교 불가능한 CPU 근거는 확정 반박이나 모든 방어의 증명이 아니다.
  짧아진 terminal line도 기존 equal-length 비교를 자동 우회하지 않는다.
- observer는 선택 policy와 `counterline_completed`를 별도로 전달한다.
  실제 C 생성은 `full_suffix_replayed=true`로 표시하지 않는다. 초기 대체 Reply의
  attempt/acceptance와 전체 C 호출·완료·소비 계수의 의미도 구분한다.
- 새 검색 identity는 `pals-restricted-refinement-post-repair-continuation/1`, 명세
  version은 `pals-post-repair-continuation/1`, policy는
  `actual_opponent_continuation_v1`이다. 468-byte conditions의 SHA-256은
  `876c7104c28183131243101df86c386865bf903f26545edde01d48de4403d4d1`이다.
  UCI driver getter·immutable startup registration·영수증·V3 lock·arena argv와
  양쪽 process snapshot을 직접 대조한다. 기존 lane의 hash/tuple은 변경하지 않는다.
- 한 번의 Reply만 소비하도록 설계된 native refinement collector 등록과 observer는
  새 lane을 명시적으로 거부한다. 일반 native UCI 역할 평가기 연결을 그 collector의
  strict witness, Query target, utility label 또는 실제 NN 실행 성공으로 승격하지 않는다.

CPU fixture는 새 뒷수순/실제 prefix/원래 첫 수/조건부 conclusion namespace,
취소·잘못된 shape·역할 예산 초과와 불완전 CPU 근거, CLI 중복·외부 checker 거부,
driver identity, manifest/receipt의 old/new tuple 치환, mock/CPU/CUDA 선언의 argv와
시작·종료 관측 치환을 검사한다. CUDA fixture는 선언/검증 코드이며 실제 GPU 실행이 아니다.
로컬 heavy CPU·GPU·실제 학습은 실행하지 않았다. 소스의 CI 인수는 정확한 새 HEAD에서
별도로 확인한다. 실제 native 다중 Reply 인수, 등록된 observed 4N 실행·strict Query
prior·whole-action cost·utility 및 paired 대국 인수는 계속 남는다.


현재 C continuation 통합의 첫 CI `aaa37fd`에서 Linux·Windows·bindings는 성공했으나,
model CPU 576개 중 기존 source review pin과 현재 Rust bytes가 달라 52 failure/3 error가
발생했다. 이 실패 run과 네 원시 job log는 보존한다. 최종 고정 Rust engine/collector
pair를 수동 대조하고 Disabled 및 기존 single-Reply branch만 literal review profile에
추가한다. 모든 이전 pair와 policy 검사를 유지한다. Python 제품 변경은 이 두 정적
allowlist의 pin 등록에 한정하며 실행·수락 알고리즘과 `_POLICY`는 동일 AST다.
독립 fixture의 literal pin과 새 lane 거부 subcase를 갱신한다. 새 다중 C policy,
별도 binary의 실행 또는 NN 물리 완료를 이 pin 갱신으로 인수하지 않는다.


### 실제 C 후속 수순의 다중 Reply native 관측 접점

새 `ActualOpponentContinuationV1`을 frozen CPU native collector까지 연결한다.
`PalsNativeContinuationRegistration`은 별도 immutable 등록 타입이며
`rz-pals-native-continuation-registration/1`과 실제 continuation policy tuple을
독립 raw hash·기준 registry·collector binary에 묶는다. 기존 refinement 등록 parser는
새 등록을 거부하고, 새 parser도 기존 등록을 거부한다. 실제 engine getter·조건 hash를
다시 대조한 뒤 선택한다. 기본 Disabled와 기존 single-Reply trace는 유지한다.

`load_cpu_with_continuation_policy`와 `pals_collect`의 명시 옵션
`--post-repair-recheck actual-opponent-continuation-v1`이 같은 경로를 사용한다.
CPU provider·strict producer·독립 등록은 실행 전에 요구한다. 기존 `--refinement-registration`
경로/해시 인수는 선택한 policy의 별도 등록 타입으로 읽는다. 새 의존성·feature·workflow·
계약 revision·탐색 의미는 추가하지 않는다.

새 책임은 `crates/rz-arena/src/pals_collect/continuation.rs`에 둔다. 기존 native owner의
prepared tensor·raw 출력·물리 완료·delivery·accepted context·producer journal을 재사용한다.
다중 Reply 각각의 actual prefix·RequestId·epoch·순서·세대·revision·원래 마감과
직전 수락 출력을 대조한다. 최초 대체 응수는 기존 선택 규칙에 따라 원래 응수와 다른
합법 수인지 확인하고, 이후 tail은 직전 actual C policy의 첫 수와 일치해야 한다.
선택 ranking의 전체 독립 재실행과 CPU endpoint의 원시 provenance 인수는 별도다.

등록·준비·각 Reply bind·종료는 `native-continuation-traces.jsonl`의 별도 domain으로
연결한다. 시작 시 repaired 길이에 따른 유한 row/byte credit을 예약하고, 취소·부분 완료의
미사용 credit도 drain까지 청구한다. 완료가 누락된 호출을 다음 prefix에 연결하거나,
이전 repaired suffix를 끼워 넣거나, 부족한 call coverage를 완전 수순으로 표시하지 않는다.
논리 취소·실제 물리 완료·출력 소비는 각각 기록한다. trace persistence와 종료 실패의
보존은 기존 Output·finish guard 경로를 사용한다.

CPU observer fixture 5개와 CLI 검사 1개를 추가했다. fixture의 물리 완료 callback은
합성 입력·출력으로 수집기 상태를 검사하며 실제 NN 실행 증거가 아니다. 기존 Python
DEFAULT/single-Reply 소비자는 literal source review profile만 추가하고 실행/수락 AST,
이전 pair, policy를 유지한다. 새 domain·등록·policy를 이 소비자의 witness로 승인하지 않는다.
현재 수정 HEAD의 CPU CI는 아직 미관측이다. 실제 frozen 다중 NN 실행과 strict 결과 인수,
Query prior·whole-action cost·utility·paired 대국은 계속 남아 있다. GPU 보류·실제 학습 제외,
로컬 heavy CPU 미실행, Draft 유지.


### 다중 Reply 최초 응답의 독립 순위 대조 — 관측 버전 2

직전 `59e57be`의 관측 접점은 CPU CI 네 job에서 인수했다. 위 최초 응답의
합법성·차이 검사에 더해, 이번 변경은 **탐색의 기존 선택 알고리즘을 바꾸지 않고**
실제 선택 근거를 native 수집기에서 독립 대조한다. 첫 C Reply 제출 전에
`RecheckPrepared.examined_responses`로 anchor의 실제 edge 목록을 전달한다.
Disabled와 기존 single-Reply의 목록은 비어 있으며 해당 관측 wire는 유지한다.
복사 범위는 현재 anchor의 합법 수로 제한하고 node/store/graph 전체는 복제하지 않는다.

수집기는 정확한 Rules replay로 목록의 합법성·중복을 확인하고 첫 raw 순위 저장 공간을
제출 전에 예약한다. 기존 physical completion callback의 유한 raw logits에서 내림차순,
동률 시 요청의 합법 수 index 순으로 순위를 만든다. 첫 호출의 결과만 pending descriptor에
보존하며 이후 tail의 전체 순위는 추가 저장하지 않는다. 논리 수락·context·producer가
완료된 호출에서 원래 응답을 제외하고 미탐색 응답 중 첫 수를 찾으며, 미탐색 응답이
없으면 다른 응답 중 첫 수를 찾는다. engine의 selected-response 값을 이 계산의 입력으로
쓰지 않는다. 실제 다음 prefix와 최종 selected response는 이 예상 수와 같아야 한다.
합법적이지만 순위가 틀린 응답은 다음 제출 전에 거부하고 종료 trace에도 성공으로
기록하지 않는다. 다른 응답이 있는데 `NoAlternativeResponse`로 표시하는 경우도 거부한다.

새 trace domain은 `rz-pals-native-post-repair-continuation/2`, observer version은
`pals-post-repair-continuation-observer/2`다. prepared에 examined 목록과 선택 규칙,
finished에 독립 예상 응답과 대조 상태를 추가한다. 이전 `/1` 원시 기록을 덮어쓰거나
새 검사에 통과한 자료로 자동 승격하지 않는다. registration `/1`과 정책 의미는 유지하며,
등록 자체는 여전히 실행 증거가 아니다. 기존 Python DEFAULT/single-Reply 소비자의
허용 source profile만 추가하고 이전 profile·policy·나머지 실행/수락 AST를 보존한다.
새 `/2` JSON의 표지만으로 strict witness나 학습 목표를 만들지 않는다.

CPU observer 검사 세 개를 추가했다. 정상 선택 네 경우는 원래 응답 제외, 미탐색 우선,
전부 탐색한 경우의 fallback, logits 동률을 다룬다. 별도 거부 검사는 합법적인 다른
순위의 수를 다음 prefix/최종 반환에 넣거나 대안이 없다고 주장하는 경우, 중복/불법
examined 목록을 다룬다. 기존 실제 engine mock callback도 준비 목록의 원래 edge·합법성·
중복을 검사한다. fixture는 합성 logits이므로 실제 NN 실행 증거가 아니다.
현재 수정 HEAD의 CI는 push 후 별도로 인수한다. frozen 실제 다중 Reply 실행·strict 결과
소비·CPU endpoint provenance·Query prior·whole-action cost·utility·paired 인수는 계속 남는다.
GPU 검증 보류·실제 학습 제외·로컬 heavy CPU 미실행·Draft 및 전체 목표 active를 유지한다.

### 외부 frozen replay의 캡처와 출력 연결

직전 `50c12dd`는 Linux·Windows workspace, bindings, model CPU CI 네 job 모두 성공했다.
이번 변경은 게임 안 다중 Reply 관측과 별개인 `rz-arena::pals_replay`의 외부 frozen 4N
경로를 다룬다. 실제 `OwnedReplayCapture`에 `bind_delivery`를 추가하고,
준비 소유자의 원래 W가 끝나기 전 취소·transport closure·postflight 상태를 검사한다.
실패는 capture를 소비하지 않으며 pending child와 입력 소유권은 기존 owner에 남는다.

CLI의 고정 `{cli,native}` 봉투에서 원본 native 바이트를 빌려 해시를 대조한다.
JSON을 다시 직렬화해서 본문 pin을 만들지 않는다. 전체 출력·header·expected 파일의
기존 한도를 유지하고 중복 key·추가 출력·알 수 없는 header와 clock 필드를 거부한다.
준비된 `/3` expected, 요청·binary·runtime·asset pin, semantic producer scope,
원래 W/E/output 선언, native 입력 audit의 parent/current/frozen/query binding을 대조한다.
CLI에 있는 `stdout_delivered=null`, process/loaded-provider/physical-closure false 표시는
그대로 요구하고 실제 pipe/exit/group 종료 증거는 capture owner에서 확인한다.
정리·직렬화가 E를 넘을 수 있다는 기존 진단은 새 실행 창이나 작업 완료 증거로 바꾸지 않는다.

작은 구조화 prior는 닫힌 필드와 단계·move 배열·크기 한도를 검사하며,
별도 `ReplayDeliveryRegistration.query_prior_source`의 독립 source digest와 대조한다.
기존 prepared `/3`에는 이 source pin이 없으므로 수신 JSON이나 현재 빌드의 digest를
기대값으로 자동 채우지 않는다. source 선언을 연결한 사실과 해당 source/NN 실행을
인수한 사실도 구분한다. production API는 실제 capture 없이는 `BoundReplayDelivery`를
생성할 수 없고 raw body 복제나 child custody 복제를 제공하지 않는다.

**`BoundReplayDelivery`는 native 의미 인수나 다음 Query prior가 아니다.**
`unadmitted_projection()`은 읽기 전용 JSON이다. native 단계 완료·Rules 수순·raw report·
역할 receipt·물리 수명 검증, strict current selector/frozen 재입장, caller chronology,
whole-action cost·utility를 이 연결 검사로 대신하지 않는다. `Unobserved` prior의 원문
연결도 가능하지만 readiness는 그대로 남으며 utility/target/training 권한은 모두 false다.
권한 타입에 Deserialize·Clone 생성자를 추가하거나 기존 Query/2 소비자를 완화하지 않는다.

이번 CPU parser 검사 다섯 개는 원문과 재직렬화의 해시 차이, borrow 범위,
raw 변조·중복 key·추가 출력, 독립 pin·clock 대체, CLI 완료/권한 주장의 거부,
에러 lane, prior/source/배열 한도를 다룬다. 합성 wire 검사는 실제 child나 ONNX 실행을
만들지 않는다. 현재 수정 HEAD의 컴파일·CI는 push 후 별도로 인수한다.
실제 등록 frozen 4N 실행과 native 의미 결과 소비·다음 Query 연결 및 유용성·paired 인수는
계속 남는다. GPU 검증 보류·실제 학습 제외·로컬 heavy CPU 미실행과 Draft를 유지한다.

### 캡처된 prior의 보고 상태와 공통 readiness 판정

직전 `7629e87`의 CPU CI 네 job과 Linux/Windows의 delivery 검사 다섯 개는 성공했다.
이번 변경은 `BoundReplayDelivery::check_reported_consistency`를 추가한다. 원래 W와 취소를
확인하고 이미 원문과 연결한 native JSON을 닫힌 필드·자료형으로 읽으며, 준비된 expected
소유권을 이동해 독립 model/export/encoding/adapter 식별과 입력 binding을 대조한다.
원문 bytes와 process custody는 기존 capture에 남는다. JSON의 중복 key 거부는 기존
delivery decoder에서 수행하고, report 소비자도 스트리밍 byte counter·4개 CPU 단계·
16KiB projection·96개 PV와 16개 line 한도를 검사한다. raw Debug 문자열은 feature로 해석하지 않는다.

기존 in-process readiness의 판정문을 읽기 전용 fact view에 대한 공통 함수로 옮겼다.
실제 native 경로는 원래 typed 관측을 빌려 사용하며 JSON 직렬화·역직렬화나 body 복사를
추가하지 않는다. caller 경로는 private wire를 통해 같은 판정문을 사용한다. window·
CPU phase/depth/report·노드 합계·endpoint·반환/소비/물리 종료 계수를 다시 대조하고,
보고된 readiness가 계산 결과와 다르면 거부한다. scope가 미완료거나 종료 관측이 없으면
기존 실패/미완료 readiness를 그대로 유지한다. 정리가 E를 넘는 경우와 작업이 E를 넘는
경우도 기존처럼 구분한다. policy·평가 계약·기본 feature·실제 역할 선택은 바꾸지 않는다.

`ReportedPriorConsistency`는 별도 readonly 타입이며 보증 범위는
`reported_consistency_pending_native_witness_and_caller_chronology`다.
**계수가 모두 맞는 합성 보고서도 실제 native witness나 다음 Query prior가 되지 않는다.**
공개 NativeRoleReceipt/ReplayQueryPrior/CheckedNativeReplayPrior에 Deserialize나
소유 projection 반환 접점을 추가하지 않는다. 작은 data enum만 역직렬화를 지원한다.
receipt 중 기존 readiness/identity에 필요한 부분만 보고 자료로 읽으며 나머지 native
상세 수명·실행 증거는 원문에 보존하고 인수하지 않는다. 별도 witness·Rules·current/frozen
재입장·caller chronology·whole-action cost·utility 검사는 여전히 필요하다.

CPU 검사 세 개를 추가했다. 기존 native 소비자와 caller의 window/부분/terminal/정리 사례
7개를 대조하고, 과장된 readiness·잘못된 자료형·미지원 필드·authority/source 대체를 거부한다.
synthetic closure의 일관성이 맞아도 reported scope와 모든 권한 false를 유지하고,
반환 계수·진행 중 실행·격리·모델 식별·실패 필드 부재가 바뀌면 거부한다.
현재 수정 HEAD의 컴파일·CI는 push 후 별도로 인수한다. 실제 등록 frozen 4N child/NN 실행,
독립 native witness·다음 Query·효용·paired 인수는 남으며 GPU 보류·실제 학습 제외·Draft를 유지한다.

### 캡처된 Repair·상대 counterline의 원래 Rules 재생

직전 `341e036`의 CPU CI 네 job과 Linux/Windows의 reported 검사 세 개·delivery 검사 다섯 개는
성공했다. 이번 변경은 `BoundReplayDelivery::check_reported_rules_consistency`를 추가한다.
이미 닫힌 보고 상태를 대조한 뒤, **이 capture가 보유한 private PreparedReplayRequest**를 빌려
원래 input artifact와 전체/정리 시간 선언을 대조한다. 임의 JSON bytes로 prepared owner를
만들지 않으며 다른 원문·다른 시간 선언의 report를 같은 Rules 결과로 받아들이지 않는다.

준비 시 확인한 root graph는 이전과 같이 준비 종료에 해제한다. 결과 검증에서는 immutable
원문을 다시 hash 확인하고 그 안의 original action/CPU `position_command`를 기존 semantic
root builder로 재생한다. 완전한 startpos 이력, 이력 상한, 원래 semantic root descriptor의
상태·이력 식별과 합법 수 순서, CPU root FEN과 ongoing 조건을 다시 확인한다. FEN만으로
과거 이력을 만들거나 native Debug 문자열에서 Position을 복원하지 않는다. 입력 준비에
사용한 original-root SemanticRequest 구성은 공통 함수로 추출하며 내용은 유지한다.

**이 재생은 결과 검증을 위한 원래 W 내 작업이다.** 원래 S를 W와 원래 whole_wall_ms에서
복구하고 original action decoder에도 그 S를 준다. E가 끝났어도 W 내 검증은 가능하지만
E를 연장하거나 새 CPU search·role/provider·model work를 생성하지 않는다. 재생 단계는
W를 확인하고, phase와 각 line ply에서는 취소를 검사한다. 임시 root/branch graph는
검증 반환 전에 해제하며 결과에는 report와 작은 Rules facts만 남긴다. source-owned
backing credit를 실제 validator/RSS peak로 승격하지 않으며 메모리 개선을 주장하지 않는다.

완료/Rules-terminal opponent endpoint 보고에 한해 다음을 Rules로 검사한다. Repair와
counterline은 원래 prefix를 유지하고, anchor 앞의 모든 수가 같으며 첫 상대 응답은 달라야
한다. anchor는 ongoing인 상대 차례여야 한다. 모든 수가 합법이고 중간 Rules terminal
뒤에 수가 이어지지 않아야 한다. 실제 counterline suffix 길이와 role/accepted 계수를
대조하며 ongoing endpoint는 선언한 line horizon까지 도달해야 한다. 종료 outcome/flag는
Rules가 재생한 실제 endpoint와 같아야 한다. 메이트 winner와 스테일메이트 draw는 Rules가
제공한 PlayStatus로 보존한다. 부분 보고는 완료 결과로 채우지 않고 원래 report/raw에 남긴다.

반환 타입 `ReportedRulesConsistency`의 범위는
`reported_lines_rules_checked_pending_repair_anchor_selection_native_witness_and_caller_chronology`다.
현재 projection에는 **Repair 이전 모델 counterline과 실제 Repair record revision이 없다.**
따라서 구조적으로 유효한 anchor라고 해도 생성자의 “자기 수 변경 뒤 첫 상대 차례” 선택을
독립 인증한 것은 아니다. 이 증거, 모든 CPU PV 의미, 독립 native witness, current/frozen
재입장·다음 Query chronology·whole-action cost·utility는 여전히 별도 인수 대상이다.
public native/projection capability에 Deserialize/Clone/소유 반환을 추가하지 않는다.
원래 Rules 복원 오류는 stage·cause를 별도 오류 variant에 보존하며, 실패 시 capture의
원문과 child custody를 취소·해제하지 않는다.

CPU 검사 일곱 개를 준비했다. 실제 private 준비 원문에서 root를 복원하고 다른 이력과
E/W 구분을 확인하며 backing 변조·취소·W 만료를 거부한다. 합법 대조선, 불법 수·prefix·
자기 차례 anchor·같은 첫 응답·잘못된 role 수·거짓 terminal·짧은 horizon·메이트 후 추가 수를
검사한다. 실제 메이트/스테일메이트는 Rules로 판정하며 native closure 미관측은 그대로
유지한다. 이번 HEAD의 CPU CI는 push 후 별도 인수한다. 실제 frozen 4N child/NN 실행,
독립 witness·다음 Query·효용·paired 인수는 남는다. GPU 보류·실제 학습 제외·Draft를 유지한다.

첫 `1eaf26a`의 CI는 test-only prepared fixture helper의 가시성이 좁아 Linux에서
`E0603`으로 실패했다. bindings job은 성공했지만 남은 model/Windows job은 이 실패를
확인한 뒤 총괄이 종료 요청해 cancelled로 보존한다. 전체 성공으로 기록하지 않는다.
helper의 `cfg(test)` 접근 범위만 strategic_action 내부로 수정하며 제품 경계·wire·동작은
바꾸지 않는다. 수정 HEAD의 새 CPU CI는 별도로 인수한다.

### 실제 Repair record와 anchor의 읽기 전용 생성자 근거

직전 `2dec62f`의 CPU CI 네 job과 Linux/Windows의 Rules/reported/delivery 검사 15개는
성공했다. 다음 변경은 `FreshReplayOwner::opponent_repair_origin`이다. 현재 owner의
accepted Repair LinePool 경로·record kind/revision·root state·원래 model counterline과
repaired line을 기존 anchor 판정으로 다시 대조한다. 저장한 anchor와 같은 “자기 수가
변경된 뒤 첫 ongoing 상대 차례”를 계산해야 실제 origin view를 반환한다.

`ReplayOpponentRepairOrigin`은 owner를 빌리는 읽기 전용 타입이다. 모델 counterline과
Repair line, 실제 record revision, anchor를 관측한다. Deserialize/Clone/소유 생성자는
없으며 local revision을 portable record ID·V feature·전략적 값으로 쓰지 않는다. 기존
실행은 원래 E의 control을 같은 판정문에 전달하고, 결과 검증은 전달받은 원래 W를
사용한다. 새 CPU 작업·model call·role 수·record·예약을 만들거나 E를 연장하지 않는다.
확인 전 anchor가 없으면 미관측으로 남기고, record/revision/line/anchor 대체는 거부한다.

CPU 검사 세 개로 실제 deterministic RoleModel+자체 CPU replay의 기록을 빌려 확인하고
추가 작업 계수가 없음을 대조하며, record 누락·revision/anchor/model line 대체 및
취소·마감 만료를 거부한다. 이것은 native NN witness가 아니다. 이 근거를 구조화하는
명시적 새 result lane·독립 source pin·caller Rules 대조와 다음 Query/실행 인수는 아직
후속 연결이다. 현재 HEAD의 CPU CI는 push 후 별도로 확인한다. 전체 목표 active·Draft,
GPU 검증 보류·실제 학습 제외·로컬 heavy CPU 미실행을 유지한다.


### 명시적 Repair origin 결과와 호출자 첫 anchor 재검산

직전 생성자 접점 `3258c823e7dc667383f3a8a84febc52b8e272863`는 CI 37882349630 attempt 1의 네 CPU job에서 인수했다. Linux·Windows 원시 로그에 새 owner 검사 3개와 이전 Rules/reported/delivery 15개의 실제 성공이 있다. 다음 코드 `8ec443dc3e66250643e78869aad46aea81732aa8`는 source split과 원래 계약을 유지하며 아래 경로를 연결한다. 이 새 source의 CI는 게시 후 별도로 확인한다.

- `prepare_repair_observed_replay_launch_bundle`과 `repair_anchor_v1`을 명시적으로 선택할 때만 native observation `/4`를 만든다. 기존 library/scoped bytes와 Query prior `/3`는 원래 경로·의미를 유지한다. 기존 `/3` 소비자는 새 evidence를 거부한다.
- 실제 `FreshReplayOwner`의 accepted Repair record를 original W에서 다시 확인하고, Repair 이전 모델 반격선·Repair 수순·local record revision·첫 anchor를 고정 크기 move projection으로 전달한다. 최대 16수이며 모델/session 생성 이전에 작은 저장 공간을 준비한다. 새로운 CPU/model/role 실행·시계 연장·graph 복제는 없다. 실패한 replay의 원래 typed cause·관측·물리 drain은 유지하며 성공한 work의 origin 검사 실패도 typed Run 오류로 남긴다.
- CLI의 source·원문 hash·요청 lane·원래 시계와 호출자의 실제 transport closure를 대조한다. Repair evidence source는 호출자에서 독립 등록한다. 수신 digest나 현재 compiled source를 production expected pin으로 대신하지 않는다. caller capture/raw stdout을 빌리므로 native body를 복제·재직렬화하지 않는다.
- 닫힌 보고 parser는 source·shape·move extent·record 존재와 revision·정책 권한을 검사한다. 기존 private 준비 원문을 빌려 Rules root/history를 한 번 재구성하고, 기존 수순 legality/terminal 검사와 첫 own 변경 다음의 **첫 ongoing opponent turn**을 같은 W에서 재계산한다. original C 응답·prefix·partial model line·후속 anchor 선택을 거부한다. parent/current/frozen/input/action/semantic/artifact/factory/scope 결합도 실제 준비와 대조한다. 임시 Rules graph는 반환 전 해제한다.
- 돌아오는 것은 `ReportedRepairRulesConsistency`라는 보고 일관성이다. local record revision은 출처 메타데이터이며 V feature·portable ID가 아니다. reported native completion을 실제 native witness로 만들거나 다음 Query/utility 권한을 발급하지 않는다.

새 CPU 검사는 첫 anchor/수순·원래 W와 취소·private 준비 binding·새/기존 lane 구분·source/shape/revision/authority 변조·raw body 보존과 CLI 4N routing을 다룬다. 형식 검사·diff check 및 common readiness predicate 원문 대조는 통과했고, 이 source의 CPU CI·실제 registered frozen child/NN·독립 native witness·current/frozen/next Query chronology·whole cost/utility·실제 multi-Reply 수집·같은 integration SHA paired 및 전체 P0–P6 최종 인수는 구분한다. GPU 보류·학습 제외·로컬 heavy CPU 미실행·PR Draft·목표 active를 유지한다.


### 호출자 관측 시계와 결과 검산까지의 비용

직전 `4c0ee9cedcbbc86bf163ce6f49f61fb5c9660574`의 네 CPU job 및 관련 검사 29개 인수 후 `4d7855925a2ca4bc2e5b115425889d90b1f3a806`에서 부모 호출자가 실제 관측한 시각을 Repair 보고의 Rules 검산 결과와 연결했다. `rz-arena/src/pals_replay/chronology.rs`는 process 감독과 native 보고 parser의 책임을 바꾸지 않고, 두 소유자의 시계를 대조하고 비용을 묶는 호출자 책임을 갖는다.

- `OriginalProcessOutput`이 실제 감독에 사용한 private process-local S/E/W를 보존한다. `checked_timing`은 stdin 전달·stdout/stderr EOF·exit·group cleanup·reap custody·오류 없는 종료와 관측 시각의 순서를 모두 요구한다. 경계 밖 시각·누락·순서 역전은 transport 완료로 인정하지 않는다. launch 시각은 `Command::spawn` 직전으로, child ready나 모델 시작 시각이 아니다.
- `OwnedReplayCapture`은 사후 파일 확인을 마친 부모 반환 시각을 보존한다. `BoundRepairReplayDelivery::check_reported_rules_with_timing`은 실제 원문 capture를 빌리고 원래 private 준비의 Rules 검사와 같은 S/E/W인지 대조한다. 검사 시작·완료는 부모 함수가 직접 찍으며 native JSON이나 외부 호출자가 시간을 공급하지 않는다.
- 준비와 사전 확인, 감독 준비, spawn부터 exit 관측까지, drain과 감독 반환, 호출자 사후 확인, capture 후 검사 전 공백, report/Rules 검산을 분리한다. 합계는 원래 S부터 이번 report 검사 완료까지이며 준비나 지연 시간을 새 시계로 제외하지 않는다. 후속 Query 선택·직렬화 및 이번 timing 확인 이후 비용은 episode 소유자가 추가로 청구해야 한다. `remaining_original_whole_ns`는 **검사 완료 시점**의 잔여량이지 사용 시점의 가용 예산이 아니다.
- 반환 객체는 실제 capture와 raw hash를 빌리고 작은 Rules 사실·시각만 보존한다. 공개 생성자·Clone·serde와 native/Query capability는 없다. 기존 `/3`·`/4` 원문 schema·digest·독립 source 등록·readiness·규칙·탐색 정책·resource 한도는 유지한다. 취소나 원래 W 만료 시 거부하고 pending child·raw bytes를 소비하지 않는다.

기존 실제 `/bin/cat` Linux 감독 검사에 원래 시계와 관측 순서·변조 거부를 추가했고, CPU 경계 검사는 postflight/check 역전·W 경계·시간 차감의 비포화와 준비 비용 유지 여부를 다룬다. 이들은 측정값이나 native 실행의 증거가 아닌 코드 정확성 검사다. 새 source의 exact-head CI는 별도로 인수한다. 실제 registered frozen 4N child/NN와 독립 native witness, current/frozen/next Query의 episode 순서, whole-action utility, 실제 multi-Reply 최종 인수와 같은 source의 paired 실행은 여전히 남는다. GPU 보류·학습 제외·로컬 heavy CPU 보류·Draft를 유지한다.


### 실제 capture의 다음 Query 자료 발급 경계

직전 `6abf6afde178a04a314283025982bcdc2cfc19da`의 CI 37887299857 attempt 1 네 CPU job, Linux·Windows 관련 검사 31개씩 및 실제 Linux process 시계 검사를 인수했다. `9c26685af7146a94550c1a998a7c134eef776df8`는 호출자 관측 비용에 원래 입력 소유자와 단일 자료 발급을 연결한다. `rz-arena/src/pals_replay/query_material.rs`는 다음 private Query에 넘길 자료의 발급·예약을 담당하며, 실제 Query 검산·native witness·utility와 구분한다.

- 실제 감독 진입 전 부모 시각을 `OwnedReplayCapture`에 보존한다. source 독립 등록, 원문 body/CLI binding, 기존 원래 Rules 검사와 실제 process 시계 검사를 재사용한다. caller 진입→supervisor 진입→report 검사→자료 반환 전 관측 순서를 같은 원래 S/E/W에서 확인한다. 시각이나 순번은 수신 JSON에서 공급하지 않는다.
- `prepare_reported_repair_prior_material`은 private 원래 `PreparedReplayRequest`와 raw stdout body를 빌려 `ReportedRepairPriorMaterial`을 만든다. parent/current/frozen/encoding 및 이전 query/catalogue/before/prior ledger/semantic 결합은 실제 원래 입력의 audit를 읽는다. body 재직렬화·원문 Vec 복제·Rules graph 보존·새 model/session·공개 생성자/Clone/serde가 없다.
- capture별 원자적 **available→checking→issued** 예약으로 동시에 두 Rules 이력을 재구성하지 않는다. 발급 전 source/parser/binding/Rules/취소 검사 실패는 예약을 해제하고 원래 W 안에서만 수정할 수 있다. 발급 뒤에는 자료를 버리거나 취소를 다시 해제해도 재발급하지 않는다. 발급 후 만료·취소로 반환이 실패한 경우에도 issued 상태를 유지한다. raw bytes·원래 준비·pending child custody를 소비하거나 정리됐다고 표시하지 않는다.
- 원래 S부터 자료 반환 전 관측까지의 비용과 report 검사 이후 추가 구간을 함께 제공한다. 잔여 budget을 새 timer로 만들지 않는다. 반환 자료의 여러 readonly 참조를 한 episode에 중복 삽입하지 않도록 **실제 next Query 소비자**의 물리 실행/원문·의미 ledger 검사가 여전히 필요하다. 이번 단일 발급은 전역 sequence나 서로 다른 capture/episode의 순서를 증명하지 않는다.

이 자료는 reported Repair/Rules와 실제 부모 transport/cost의 연결이다. `CheckedNativeReplayPrior`·admitted Query·reward/target은 발급하지 않으며, 실제 registered frozen 4N child/NN·독립 native witness·다음 Query admission/episode chronology·whole-action utility·최종 multi-Reply·same-source paired 인수는 남는다. 네 CPU 검사는 단일 발급, 동시 alias, failed-check 예약 반환, 취소/만료와 재발급 금지를 다룬다. 이들은 actual frozen NN/Query 실행의 증거가 아니다. 기존 native `/3`·`/4` schema·canonicalization·readiness·탐색·평가 의미·resource 한도를 유지한다. 새 exact-head CI는 게시 후 별도 인수하며 GPU 보류·학습 제외·로컬 heavy CPU 보류·Draft·goal active를 유지한다.

### 원본 입력 전달 전 실제 loaded executable 관측

원본 입력을 받는 Linux supervisor는 첫 stdin byte를 보내기 전에 자신이 소유하고 아직 reap하지 않은 실제 child PID의 `/proc/<pid>/exe`를 연다. 이 파일의 device/inode를 실행에 사용한 pinned executable FD와 비교한다. 경로·argv·자식 JSON의 자기 보고나 별도로 입력한 PID로 이를 대신하지 않는다. 열기·동일성의 미확정 관측은 아래 시작 단계의 원래 E 안에서 보류하며 확인 전 입력을 보내지 않는다. metadata 오류·내용 검증 거부나 끝내 확인되지 않은 동일성은 기존 process-group drain·종료·pending custody 경로로 실패를 보존한다.

- 성공 관측 시각과 첫 실제 pipe write 후의 부모 관측 시각은 원래 S 기준으로만 저장한다. 첫 관측과 이후 매 write 직전에는 cancellation과 원래 E를 검사하며, OS 관측 시간이 길어졌다고 timer를 다시 시작하지 않는다. OS call을 preempt하는 hard-timeout 증거로 해석하지 않는다.
- `OriginalLoadedImage`는 실제 `OriginalProcessOutput`을 빌리는 readonly 자료다. 성공 transport·동일 PID·launch→loaded-file 관측→첫 write→exit 관측 순서가 있어야 제공한다. 공개 생성자·Clone·serde가 없으며, legacy `rz-original-process-transport/1` JSON에는 새 private 관측을 추가하지 않는다. complete transport나 serialized timestamp만으로 이 borrow를 만들지 않는다.
- 단일 `ReportedRepairPriorMaterial` 발급은 같은 capture의 loaded-file 관측도 요구하고 빌린다. 실패는 발급 전 예약을 반환하고 raw/input/pending custody를 유지한다. 이 관측은 모델·직렬화 버퍼·raw output 복제를 추가하지 않는다.
- 관측 범위는 해당 순간의 executable **파일 동일성**이다. 바이너리 내용·동적 library·계속 같은 executable로 실행됐음·fork 탈출·NN 완료·cgroup enforcement·native witness·전역 Query 순서의 증거는 아니다. 기존 독립 source pin과 pre/postflight를 대체하지 않는다. child가 입력 전에 exit하거나 proc 접근을 허용하지 않는 환경은 unobserved failure로 남긴다.

실제 Linux `/bin/cat` 입력 전달 검사는 device/inode·입력 전 관측 순서·완전 종료·legacy JSON 보존과 결손/순서/PID 거부를 검사한다. 별도 Linux 검사는 실제 proc executable과 다른 pinned file의 불일치를 거부한다. CPU CI가 실제 frozen 모델 실행·Query admission이나 GPU 인수를 대신하지 않는다. GPU 보류·학습 제외·로컬 heavy CPU 보류·Draft·goal active는 유지한다.

### 실제 loaded executable 내용과 원래 실행 예산의 연결

동일 파일 관측만으로 내용 hash가 검증됐다고 표시하지 않는다. PALS의 실제 caller는 기존 바이너리 preflight에 더해, 부모 감독자가 열어 유지하는 실제 child의 proc executable FD를 원본 stdin 전에 검증한다. 기존 `read_at` 기반 16KiB 스트리밍 hash·크기·metadata 전후 대조·각 block의 원래 E/취소 검사를 재사용한다. 고정 바이너리 FD의 이름이나 자식의 자기 보고를 해시 대상 대신 쓰지 않는다. 일반 original-input API는 기존 파일 동일성 경로를 유지하고 이 추가 검증 접점은 crate 내부에 둔다.

- 시작 전에는 **바이너리 두 pass와 publication 두 pass**의 실제 bytes 합계를 checked arithmetic으로 계산해 선언된 verification read credit에 들어가는지 확인한다. 전체 read credit·최대 바이너리 크기·S/E/W·worker·output 한도를 자동 확대하지 않는다. 기존 정책이 큰 바이너리의 추가 pass를 수용하지 못하면 시작 전에 예산 부족을 거부한다.
- 내용 검증 후 원래 E를 넘었거나 취소됐으면 stdin을 보내지 않는다. 검증 거부는 이미 시작한 child를 버리는 새 `Err` 경로로 전환하지 않는다. 기존 bounded drain·group 종료·reap·pending custody와 `process.loaded_image_verification_refused`를 남기고, `OwnedReplayCapture::loaded_binary_error`에 원래 hash/크기/read-credit/clock 오류를 JSON postflight 오류와 별도로 보존한다.
- `CheckedReplayLoadedBinary`는 실제 capture의 private 내용 관측과 같은 PID의 complete process 관측을 빌린다. 등록한 binary artifact, 실제 file device/inode, loaded-file 관측→검증 시작→검증 종료→첫 write 순서가 일치해야 한다. 공개 생성자·Clone·serde나 generic callback 성공만으로 이 자료를 만들 수 없다. `binary.rs`는 이 caller-side 내용 관측의 readonly 접점이며 모델·native 결과·Query를 검산하지 않는다.
- 단일 `ReportedRepairPriorMaterial`은 같은 capture의 내용 검증을 요구하고 이 작은 관측을 빌린다. 원래 raw body·입력·부모 시각·예약·실패 반환 조건을 유지한다. 바이너리 전체 buffer·모델/session·Rules graph를 추가 복제하지 않으며 기존 transport/1 및 native /3,/4 형식을 바꾸지 않는다.

독립 인수는 구분한다. 작은 Linux actual-process 검사는 실제 cat executable의 올바른/틀린 SHA-256, cursor 보존, 읽기 credit 차감, 원래 입력 전 거부·0 byte 전송·완전 종료를 확인한다. 추가 supervisor 검사는 성공 callback 직후의 취소와 E 경과도 입력을 허용하지 않는지 확인한다. CPU 경계 검사는 read-credit overflow/부족과 검증 시각의 역순을 다룬다. 이 검사는 actual registered frozen 4N 모델/NN, runtime dependency, 지속적인 executable 동일성, hostile write·fork 격리, next Query episode/ledger·whole utility 또는 paired 대국의 증거가 아니다. 해당 native 인수와 실행 연결은 남기며 GPU 보류·학습 제외·로컬 heavy CPU 보류·Draft·goal active를 유지한다. 새 exact-head CPU CI는 게시 후 실제 로그로 별도 인수한다.

### 동일성을 확인하기 전의 유한 시작 단계

`5e0ebc7`의 CI [37893898805](https://github.com/daejunnom/RoveZero/actions/runs/37893898805)는 Linux 세 검사 실패와 나머지 세 job 성공으로 보존한다. 진단 메시지만 추가한 `b44ea69`의 CI [37895010893](https://github.com/daejunnom/RoveZero/actions/runs/37895010893)는 Linux 한 검사 실패와 나머지 세 job 성공이다. 두 번째 실패에서는 actual child의 `process.loaded_image_differs_from_pinned_file`, 0 byte 입력, 내용 verifier 미호출·read credit 미차감, group Gone·pending child 없음이 관측됐다. `spawn` 직후의 최초 불일치는 확인했지만 정확한 커널 원인이나 일시성은 이 로그만으로 확정하지 않는다.

- 시작 관측은 같은 unreaped child의 기존 supervisor loop에 둔다. proc executable 열기 실패·파일 동일성 불일치 두 종류만 원래 E와 취소 안에서 보류한다. child를 다시 spawn하거나 새로운 대기 시계·예산을 만들지 않는다. pipe drain·출력·child 수·전체 W·종료 책임도 계속 검사한다.
- **동일성 확인과 선택적 내용 검증이 모두 끝나기 전에는 write가 불가능하다.** 원래 E 만료·취소·조기 exit·지속 불일치는 입력 없이 종료하고 `process.loaded_image_not_confirmed_before_input` 및 마지막 보류 사유를 보존한다. metadata·비정규 파일·시계·관측 횟수 overflow는 즉시 실패한다. 내용 hash·크기·read-credit 검증 거부는 재시도하지 않는다.
- 관측 횟수·보류 횟수·첫/마지막 보류 코드는 실제 process owner의 작은 private scalar로 보존하고 readonly getter로 제공한다. 성공 후 이전 보류 관측을 지우거나 성공 receipt의 오류로 바꾸지 않는다. legacy transport/1 JSON은 그대로 유지한다. 관측 횟수는 syscall의 hard-timeout·자식의 ready·NN 완료 증거가 아니다.

실제 cat 입력 검사에 관측 횟수와 원래 입력 전 완료 조건을 연결하고, 직접 다른 executable을 확인하는 기존 거부 검사를 유지한다. 보류 대상이 metadata·내용 검증 오류로 넓어지지 않게 CPU 경계 검사도 추가한다. 이 source 수정의 정확한 CI는 게시 후 별도로 확인하며, 실제 frozen NN·next Query·utility·paired 인수는 계속 남긴다.
