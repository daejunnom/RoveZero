# rz-position — A 역할 규칙·상태 구현

`TASK-A01/A02`의 표준 체스 규칙과 `TASK-A03`의 concrete state 경계를 자체 Rust 코드로
구현한다. 제품 의존성은 없고 신규 코드는 [MIT](LICENSE)다. 기준 지시서는
[IMPLEMENTATION-DIRECTIVES](../../docs/IMPLEMENTATION-DIRECTIVES.md), 의미 계약은
[CONTRACTS](../../docs/CONTRACTS.md)다. 연구 `CARD-A*`와 담당 `TASK-A*`는 다르다.

## 제공하는 동작

- `Position::startpos/from_fen`: 64칸·bitboard, 차례·권리·raw EP·u32 카운터를 보존한다.
  FEN은 여섯 필드, 기물·왕·승격 재료·공격 상태·EP 직전 double push의 국소 모순을
  거부한다. 전체 과거의 도달 가능성을 증명하는 retrograde 판정기는 아니다.
- `legal_moves/ordered_legal_moves`: 왕 포획 없는 완전 합법 집합을 제공한다. 공격
  판정은 pinned piece의 공격을 포함하고, 캐슬링 통과 칸과 EP 발견 공격을 검사한다.
  a1=0/h8=63이며 순서는 `(from, to, promotion)`이다. 같은 from/to 승격은 Q/R/B/N 순이다.
- `make_move/make_uci/make_from_view/unmake`: checked 전이와 단일 사용 undo를 제공한다.
  `RuleMoveDelta`는 기물 제거·추가, capture/castling/EP/promotion의 조합, before/after
  불변 상태를 담는다. 실패·overflow·외부 owner·잘못된 child는 원본을 보존한다.
  `apply_uci_moves`는 전체 입력열을 원자적으로 적용한다. 성공한 입력 transaction은
  mutable owner를 교체하며 이전 live view/undo를 무효화한다.
- `classify_position`: 진행/종료, 청구 근거, 이력 근거를 독립 필드로 반환한다.
  3회/100 ply는 현재·예정 수의 청구 조건이고 5회/150 ply는 자동 종료다.
  마지막 수의 mate가 150 ply 자동 무승부보다 우선한다. Search/Arena는 분류를 먼저
  소비하여 대국 종료와 claim policy를 적용한다. make/perft 자체는 board legality를 따른다.
- `snapshot`: `Send + Sync`인 소유 불변 view다. 이후 live 상태 전이와 독립적이며
  newest-first `known_history`를 읽을 수 있다. 이력 노드는 공유 prefix를 유지하고
  긴 unique prefix는 반복으로 해제한다. 전체 Debug는 이력을 요약한다.

## 이력·identity·수명

startpos의 이력은 complete, FEN 이전은 unknown_prefix다. Unknown history를 fill해
실제 과거로 기록하지 않는다. 알려진 3/5회 발생은 prefix가 미상이어도 입증 가능하다.
알려진 pawn move/capture/권리 소실 뒤에는 그 이전 반복이 불가능하므로 반복 증거만
complete로 판단할 수 있다. 원본 게임·모델 이력의 unknown_prefix는 유지한다.

`RepetitionIdentity`는 차례·기물·권리·실제 합법 EP 가능성만 비교한다.
`PositionIdentity`는 raw EP·카운터·전체 알려진 이력·origin/completeness를 정확히 비교한다.
모델 입력 cache identity는 C/D가 실제 encoding·history-fill·weight·precision과 함께
구성해야 한다. 이 규칙 identity를 모델 cache key로 바로 사용하지 않는다.

`snapshot.same_state`는 의미 상태의 복원 비교다. Live revision은 make와 unmake마다
증가하고 되돌리지 않는다. Live view 적용에는 owner·revision·정확한 history node를
모두 검사하여 오래된 상태와 ABA 재사용을 거부한다. Root/game generation과 deadline은
총괄 공통 계약 및 B/D의 별도 소유이며 position revision으로 대신하지 않는다.

## 지원 프로필과 유한 자원

`RULES_VARIANT=standard_chess`, `RULES_VERSION=rz-position/0.1.0`이다.
Dead position은 `proven_material_subset_v1`: K/K, KB/K, KN/K, 왕·동색 칸 bishop만 있는
상태를 exact로 입증한다. KNN/K·KN/KN·반대색 bishop 등과 일반적인 모든 dead position의
완전 판정은 제공하지 않는다. 첫 대국 전에 총괄/E가 이 범위와 runner 판정을 잠가야 한다.
Claim 자동 수락·tablebase·평가 adjudication은 이 crate의 규칙 결과와 별도 정책이다.

`PositionLimits` 기본값은 FEN 4,096 bytes, 알려진 이력 4,096 positions, perft depth 8,
전체 traversal 50,000,000 nodes다. 이력 상한은 조용한 절단 대신 명시 오류를 반환한다.
Perft의 configured depth는 스택 안전 상한 64 이하이며, perft/divide는 내부·terminal
nodes도 같은 전체 노드 예산으로 센다. 카운터·revision overflow는 명시 실패다.
진행 상태에서 intended claim preview를 위한 자원이 부족하면 classification도 오류를
반환한다. 확정 terminal은 preview 없이 판정한다.

## 빌드·독립 검증

검증한 compiler는 Rust 1.90.0, CPU Linux x86_64다. 총괄 workspace가 아직 없어
manifest 경로로 독립 빌드한다. 임시 standalone lockfile은 crate ignore에 두며
공유 workspace 등록·root lockfile·toolchain·CI는 TASK-I01 소유다.

```sh
cargo fmt --manifest-path crates/rz-position/Cargo.toml --check
cargo clippy --manifest-path crates/rz-position/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path crates/rz-position/Cargo.toml
cargo test --release --manifest-path crates/rz-position/Cargo.toml --test perft -- --include-ignored
```

공개 [CPW perft](https://www.chessprogramming.org/Perft_Results)의 6개 FEN과 depth 1~4
golden counts를 fixture에 고정한다. 기본 검사는 depth 3 및 초기/EP depth 4,
확장 검사는 모든 depth 4와 초기 divide다. 특수 수·동일 보드/다른 이력·청구/자동 종료·
원자성·overflow·stale view·긴 undo·8,000수 작은 스택 해제도 검사한다.

독립 참조는 test-only subprocess의 `python-chess==1.999`, `chess==1.11.2`다.
외부 GPL-3.0-or-later 구현을 제품 소스에 복사하거나 연결하지 않는다. 라이브러리와
기본 테스트는 Python을 요구하지 않는다. 검사 환경만 저장소 밖에 준비한다.

```sh
: "${RZ_A_OUTPUT:?저장소 밖 검사 output root를 지정하세요}"
python3 -m venv "$RZ_A_OUTPUT/oracle-venv"
"$RZ_A_OUTPUT/oracle-venv/bin/python" -m pip install python-chess==1.999 chess==1.11.2
RZ_CHESS_PYTHON="$RZ_A_OUTPUT/oracle-venv/bin/python" \
  cargo test --release --manifest-path crates/rz-position/Cargo.toml -- --include-ignored
```

Seed 20261003, 8 games×96 ply, 공개·특수 fixture를 고정하며 현재 참조 trace는
23 cases/895 positions, 모든 초기 successor 274개, depth-3 divide 178개, 실제 trace
이동 872개다. 합법 집합·check·mate/stalemate·6필드 FEN·전이·긴 전체 이력 undo를 대조한다.
Process deadline 30초, stdout 8 MiB, stderr 64 KiB로 제한한다. 첫 불일치 FEN·move/path·seed를
보고하고, 원시 trace/log는 외부 output root에 보존한다. Mock/GPU/Elo 증거와는 별도다.

## TASK-A03 인계와 선행 항목

불변 state·raw history, ordered move view, 네 승격, exact order 비교, typed failure,
owner/revision 검증 및 cross-thread 수명을 제공한다. `BoardMove`는 A의 concrete
좌표 입력이며 shared Move ABI가 아니다. Shared ID/error/request/result 선언은 복제하지 않았다.

총괄의 `rz-contracts` Rust revision이 아직 게시되지 않아 아래 통합 연결은 미실행이다.
다른 담당의 WIP는 별도 PR로 개발되며, 실제 통합은 같은 공통 revision을 기준으로 인수한다.

1. 총괄의 primitive·handle·공통 오류를 이 concrete state/view에 adapter로 연결한다.
   필요 공통 타입 재배치는 총괄 revision으로 수행하고 이 local API를 frozen ABI로 취급하지 않는다.
2. B/C와 legal policy gather/scatter·승격·permutation·side-to-move WDL·terminal 선행을 대조한다.
3. B/D와 request/game/root generation·deadline·늦은 결과·새 position의 경계를 통합 검사한다.
4. E/F와 raw 이력 복원·unknown prefix·claim/dead profile을 맞추고 총괄 I02가 인수한다.

A01/A02 소스·CPU 정확성 검사 완료와 A03의 실제 공통 계약/소비자 통합 완료를 구분한다.
원격 feature PR의 동일 SHA·계약 revision·실제 명령·결과로 인계하며 최종 통합은 총괄이 확인한다.
