# RunPod 이전 의미 보존 최적화 인수

기준일: 2026-10-03. 총괄 TASK-I02가 A03/B 접점을 수동으로 맞췄다.
기준 소스는 `e661a11593609405d661e7f90a236876d64b61b5`, 최적화 소스는
`1cdd707922dcaaf234b4cd162210ce9f2fa512bb`이며 [PR #16](https://github.com/daejunnom/RoveZero/pull/16)에서 공유한다.
후속 문서 head와 제품 실행 소스를 구별한다. 공통 계약은 `0.1`이고 기존 GPU 인수는
그 소스·장비의 이력으로 보존한다. [외부 GPU 계획](RUNPOD-BENCHMARK-PLAN.md)을 따른다.

## 바꾼 경로와 보존 의미

1. `ContractPosition::export`는 합법 수를 한 번 생성하고 같은 Rules 소유 배열로
   classification과 ordered view를 만든다. 외부에서 합법 배열을 주입하는 API를 만들지 않았다.
2. `RulesSearchPosition::play`는 부모를 다시 export하지 않는다. Rules의
   `fork_from_view`가 실제 owner/revision/history·합법 수 포함 여부·exact terminal을 확인하고,
   registry의 새 owner를 받은 자식만 만든다. frozen 부모·형제의 이력과 live view는 보존한다.
   다른 owner의 view·make/unmake 뒤의 stale view·같은 owner 재사용·불법 수·terminal을 거부한다.
3. claim/search preview는 undo 정보를 소비하지 않으므로 64칸 변화 스캔·두 delta 배열·
   버려지는 UndoToken을 생성하지 않는다. 실제 make/unmake와 같은 checked transition을
   사용하고 history·revision·counter 오류는 계속 반환한다. 실제 mutation의 delta·undo는 유지한다.
4. 컴파일된 불변 `IDENTITY_PROFILE`의 SHA-256만 `OnceLock`으로 한 번 계산한다.
   상태 이력·legal order digest의 바이트 framing은 바꾸지 않았다. 평가 cache/visit를 추가하지 않았다.

공통 타입·ID·revision·a1 좌표·승격 순서·side-to-move WDL·규칙 claim/자동 종료·이력
unknown 의미는 그대로다. 새로운 탐색·모델·정밀도·warm state·근사 재사용을 도입하지 않았다.
실행 시간이 줄면 시간 제한 탐색의 완료 traversal 수는 달라질 수 있으므로 같은 방문 예산의
정확성 검증과 실제 벽시계 성능 검증을 구분한다.

## CPU 측정

외부 probe는 실제 `ContractPosition::export`와 `RulesSearchPosition::play`를 호출한다.
startpos, Kiwipete, 앙파상, 승격, 20-ply 이력의 **5 root·120 edge**를 고정했다.
루트 준비는 측정 밖이고 같은 edge 순서·150회 반복·동일 release toolchain `1.96.0`을 쓴다.
probe의 별도 lock에 사용된 의존 버전은 root lock의 대응 패키지와 전부 일치했다.
root Cargo.lock SHA-256은 `e86e6deca01f9f5d74a11014cd5473dd98f6759a64540ca26377a6cb44c7b57d`다.

시간 측정은 **카운터 없는 System allocator**, 단일 CPU affinity 0, 9개 교차 block이다.
각 block에서 두 실행 파일의 순서를 바꾸며, process마다 15초·가상 주소 공간 512MiB 상한을 둔다.
호스트는 Linux 6.6.87.2 WSL2/glibc 2.39다. 이 가상 주소 한도를 실제 RAM hard cap으로
표현하지 않는다. 별도 counting-allocator probe의 9회 결과에서 할당 요청 횟수·누적 bytes를 얻었다.

| 실제 CPU 호출 경로 | 이전 | 최적화 | 같은 block의 시간 감소 중앙값 |
|---|---:|---:|---:|
| Rules export | 10.589µs/호출 | 6.243µs/호출 | 41.3% |
| Search child fork | 29.761µs/호출 | 8.132µs/호출 | 72.7% |

표의 시간은 **block 평균 시간/호출의 9개 중앙값**이다. 개별 요청의 P50/P95/P99가 아니다.
같은 block의 fork 시간 비율은 0.2672~0.2809, export는 0.5262~0.6290이었다.
fork의 평균 할당 요청은 395.708→129.625회/호출(**67.2% 감소**), 누적 요청 bytes는
34,046.083→15,134.125 bytes/호출(**55.5% 감소**)다. 누적 할당을 live RAM/VRAM peak로
해석하지 않는다. GPU·신경망·전체 탐색·대국 강도의 감소율로 확장하지 않는다.

| System probe | SHA-256 |
|---|---|
| 이전 실행 파일 | `92aa33100ca63e62093a245d3c34906df4123f404aaaa4e232c6036489103f8c` |
| 최적화 실행 파일 | `13b507586ea4f5b3eee5cadaff63d3bbdd27f00c4a88f58e7df6b61f278825de` |
| 전체 상태 witness | `76023f6da4fe21f79f656f4d3ffeb48e880c7df5935cd948820564a4a52f13c4` |

18개 process의 witness를 이전 witness와 **바이트 단위로 대조**했다. 각 root/child의
semantic·revision·legal order/배열·전체 알려진 이력 FEN·claim/terminal classification·
exact WDL을 포함한다. registry owner는 실행마다 새로 발급해야 하므로 공통 상태 비교에서
제외하고 별도 검사에서 fresh/foreign/stale를 검증한다. digest 일치만으로 바이트 대조를 대신하지 않았다.

## 검사와 인수 경계

- 최적화 소스의 workspace all-targets/all-features CPU 검사: **675 passed / 0 failed /
  16 ignored**, 65개 test binary. 실제 GPU 등 별도 환경 검사 16개는 미실행이다.
- Rules 검사 42개와 default-feature native CLI 거부 경로 4개 통과, fmt·strict clippy 통과.
  기존 독립 SHA-256 golden·특수 수·긴 make/unmake·counter/history/revision 오류를 포함한다.
  새 fork는 기존 clone→export→make→export 경로와 특수 수·claim·다른 이력을 대조했다.
- [CI run 37093554114](https://github.com/daejunnom/RoveZero/actions/runs/37093554114)는
  같은 `1cdd707`에서 Ubuntu·Windows 모두 성공했다. workflow는 위 CPU 검사와 별도로
  release extended perft·pinned python-chess/chess 대조·학습 fixture 검사를 실행한다.
- 후속 통합 product source `b5ba853585cb5a78f81159f733a86cdfe085936b`에서 실제 Maia/ORT
  CPU 수치, A Rules→C CUDA→D, CPU/CUDA UCI와 제한된 CUDA pair를 로컬 RTX 4050으로
  새로 인수했다. C CUDA raw-logit/batch 검사는 같은 SHA의 기존 실행을 입력·binary·report·
  placement hash로 재사용 확인했다. 범위·실패 이력·회수 근거는
  [로컬 연결 인수](../INTEGRATION-STATUS.md#최적화-소스-b5ba853의-로컬-cpucuda-연결-인수)를 따른다.
  이전 `e45dd105` 결과를 새 실행으로 옮기지 않았다. 후속 `88ea26e`의
  [D02 host-source 인수](LOCAL-D02-PROFILE.md)에서 탐색 요청 journal·P50/P95/P99를 실제로
  연결했다. startup warm-up을 포함한 전체 process journal·device transfer/kernel·process peak VRAM·
  통제된 GPU 최적화 A/B·강도는 별도 인수다. RunPod에서는 그 환경의 실제 지원과
  같은 비교 입력을 별도로 인수하며 기존 ignored 검사 전체가 통과했다고 표현하지 않는다.

## 중복 조사에서 유지한 경계

| 경로 | 확인한 동작 / 처리 |
|---|---|
| A→B fork/export/claim preview | 실제 소비 hot path의 중복 생성·분석을 위와 같이 제거하고 측정함 |
| C input key / prepare | 입력 key 생성과 admission의 재인코딩/identity 검증은 별도 경계다. 이 검사를 생략하지 않음; 전체 encoding 비용은 D02에서 측정 |
| D pending / scheduled request | 공통 frozen request와 physical bind를 공유하는 Arc clone이다. 수명·취소·예약을 별도 소유하는 구조를 deep tensor 복제로 취급하지 않음 |
| C runtime pin / E native launch | stream hash·새 inode·닫힌 writer·readonly pin·loaded origin은 기밀성과 별개인 입력 재현/수명 검증. hardlink·무검증 shared path로 복제를 대체하지 않음 |
| 외부 보존 | packed tar 중복은 전송하지 않고 원본+provenance를 한 archive로 보존. 동일 witness는 원시 표본의 논리 참조와 같은 bytes/hash를 확인해 한 객체로 보존 |

모든 `.clone()`을 병목으로 판정하거나 모든 파일 검증을 하나로 합치지 않았다.
새 GPU의 cold bootstrap·encoding·queue·transfer·실제 완료·backup 전체 journal을 통해
남은 비용을 확정한다. 이 CPU 최적화를 D03 GPU 전체 개선이나 LC0 강도 향상으로 보고하지 않는다.

## 외부 자료 보존

Windows 외부 `reports/coordinator-integration/pre-runpod-optimization`과 Linux 작업 전용
출력 root에 probe 소스·lock·recipe·raw TSV·witness·검사 로그를 보존한다. Git에는 요약만 둔다.
Oracle 논리 repository `oracle-artifacts/coordinator-20261003`에는 D/E/F 원본 719개와
역사적 E script 2개, 출처 receipt 4개를 보존했다. 원본 payload는 **721개·3,060,125 bytes**다.
각 파일의 원래 Git blob·SHA-256·길이를 서버에서 재검사했고 archive SCP 왕복을 확인했다.
과거 script를 새로 실행한 검사가 아니며 원시 결과의 과거 누락·fixture 제한은 그대로다.

archive는 **435,710 bytes**, SHA-256은
`52601f14ea9a03a0c1607a3db0799037379414f90ea2c0fc44777d2eb31d0587`이다.
Oracle IP·개인 key 경로·credential 내용은 공유 소스에 넣지 않았다.
사용자가 등록할 Pod Env와 Pod→Oracle 연결은 아직 실행하지 않았다.
