# rz-position — A 역할 규칙·상태 구현

`TASK-A01/A02`의 표준 체스 규칙과 `TASK-A03`의 공통 계약 적용을 자체 Rust 코드로
구현한다. 기본 규칙 코어는 의존성 없이 동작하며, 선택 `contracts` feature는 총괄의
`rz-contracts` 0.1과 `sha2=0.10.9`를 사용한다. 신규 코드는 [MIT](LICENSE)다. 기준 지시서는
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

검증한 compiler는 Rust 1.90.0, CPU Linux x86_64다. A 작업 branch는 총괄 PR #7의
루트 변경을 포함하지 않으며 manifest 경로로 독립 빌드한다. 임시 standalone lockfile은
crate ignore에 두며 공유 workspace 등록·root lockfile·toolchain·CI는 TASK-I01 소유다.
선택 의존성의 metadata는 첫 Cargo 해석 시 내려받을 수 있지만, 기본 build에서는
공통 계약과 SHA-256 구현을 컴파일하거나 규칙 코어에 연결하지 않는다.

```sh
cargo fmt --manifest-path crates/rz-position/Cargo.toml --check
cargo clippy --manifest-path crates/rz-position/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path crates/rz-position/Cargo.toml
cargo test --release --manifest-path crates/rz-position/Cargo.toml --test perft -- --include-ignored
cargo clippy --manifest-path crates/rz-position/Cargo.toml --all-features --all-targets -- -D warnings
cargo test --manifest-path crates/rz-position/Cargo.toml --features contracts --test contracts
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
  cargo test --release --manifest-path crates/rz-position/Cargo.toml --all-features -- --include-ignored
```

Seed 20261003, 8 games×96 ply, 공개·특수 fixture를 고정하며 현재 참조 trace는
23 cases/895 positions, 모든 초기 successor 274개, depth-3 divide 178개, 실제 trace
이동 872개다. 합법 집합·check·mate/stalemate·6필드 FEN·전이·긴 전체 이력 undo를 대조한다.
Process deadline 30초, stdout 8 MiB, stderr 64 KiB로 제한한다. 첫 불일치 FEN·move/path·seed를
보고하고, 원시 trace/log는 외부 output root에 보존한다. Mock/GPU/Elo 증거와는 별도다.

계약 적용 후 CPU Linux/Rust 1.90.0에서 기본 debug 29개 통과(외부 참조·확장 perft
2개 명시 ignore), `--all-features` 전체 release는 doctest 포함 **41개 통과, 0 실패/ignore**를
확인했다. 이 중 실제 공통 계약 consumer 검사는 9개다. Fmt와 전체 feature/target Clippy
`-D warnings`도 통과했다. 별도 PR #7 source checkout의 `cargo test --locked` 6개와
fmt/Clippy도 같은 CPU/compiler에서 통과했으며 원격 CI 결과로 표현하지 않는다.

## TASK-A03 공통 계약 0.1 적용

[PR #7](https://github.com/daejunnom/RoveZero/pull/7)의 실제 Rust 계약 revision 0.1을
검토하고 source `ae7bf5c20c3acdc12ef7e20aa1a88b5853ee99c8`에 고정했다. 검토 시 Draft/open,
base `develop@9f0bc598f6b2d8f863fd46af6a4fd73bfef1f0b8`, 미통합이며 해당 head의 원격
check run과 review는 없었다. 계약 게시·CPU 검사·workspace 통합은 서로 다른 인수다.

`contracts::ContractPosition`은 실제 registry가 발급한 `OwnerId`와 `Position`을
소유한다. 각 인스턴스·fork·새 게임에는 발급자가 재사용하지 않는 ID를 배정해야 한다.
Position의 mutable 참조와 wrapper Clone을 제공하지 않아 외부에서 revision의 의미를
바꾸지 못한다. `export`는 caller의 digest·classification·수 배열을 입력으로 받지 않는다.

- `ContractState::snapshot()`은 공통 `PositionSnapshot<RulesState>`다. Concrete payload의
  `snapshot()`에는 원시 상태·이력이, `classification()`에는 origin·알려진 반복 수·
  반복 증거 완전성·5회 반복 unknown 여부 등 상세 근거가 남는다.
- `ContractState::legal_moves()`는 같은 상태의 정확한 순서와 네 승격을 보존한다.
  `BoardMove`와 공통 `Move` 사이의 `TryFrom`은 좌표·승격을 변환하며 합법성은 checked
  Rules 전이가 검증한다. 공통 generic 생성자 자체는 체스 상태의 진위를 보증하지 않으므로
  소비자는 A가 export한 bundle의 metadata와 배열을 사용한다.
- `make_from_view`는 원본 concrete owner/revision/history와 합법 수를 확인한다.
  Make→unmake로 의미 상태를 복원해도 예전 view는 stale다. 오류는 공통 code/stage/detail로
  전달하며 자원·카운터 실패를 무승부로 숨기지 않는다.
- Exact terminal은 추가 대국 수와 평가 요청에서 거부한다. `terminal_wdl`은 frozen
  side-to-move 관점의 정확한 승/무/패다. Current/intended claim은 ongoing과 공존한다.
  Claim을 실행할지는 B/E의 정책이며 adapter가 자동 무승부로 만들지 않는다.

SHA-256의 고정 profile은 `IDENTITY_PROFILE` 문자열이며 `profile_digest()`로 읽는다.
State 입력은 domain의 u64-LE 길이 framing, profile digest, origin/completeness 각 1 byte,
알려진 이력 길이 u64-LE, newest-first의 각 irreversible byte와 canonical FEN framing이다.
Legal-order 입력은 별도 domain framing, profile digest, 배열 길이 u64-LE, 순서대로
`from/to/promotion` 각 1 byte(승격 none/Q/R/B/N=0/1/2/3/4)다. Owner·live revision은
semantic digest와 별도다. Export 시 전체 알려진 이력을 hash하므로 비용은 이력 길이에
비례하며 현재 adapter에 hot-path 성능 개선을 주장하지 않는다. Live 적용은 hash 일치만으로
허용하지 않고 concrete 상태를 확인한다. C/D는 실제 모델·encoding·fill·정밀도의 평가
입력 key를 별도로 만들며 repetition identity나 이 semantic digest로 대체하지 않는다.

`tests/contracts.rs`는 실제 A 상태와 공통 `EvalRequest<RulesState>/EvalOutput`을 연결한다.
CPU mock의 독립 literal 합법 순서·승격·digest fixture, full history·raw EP·origin 구별,
make/unmake/drop/thread 수명, state/order 혼용과 위조 순서, claim/terminal, side-to-move WDL,
game/root/model/encoding/backend 교체·정확한 deadline 경계·취소를 검사한다. 모델/encoding
manifest·input key·clock은 명시 mock이며 실제 C mapping이나 D scheduler를 인수한 것은 아니다.

총괄 workspace 인수에서는 pinned Git 의존성을 `path = "../rz-contracts"`로 바꾸고
root lockfile을 확정해 **단일 Cargo source의 계약 타입**을 사용한다. 같은 SHA라도 Git과
path package를 함께 사용하면 Rust 타입이 다르다. 총괄은 선택 feature·동일 계약 source로
A03 검사를 다시 실행하고 B/C의 실제 model policy mapping, B/D의 late-result와 수명,
E/F의 history/claim/dead profile을 연결해야 한다. 공통 타입·root/CI 변경은 이 PR에 없다.

A01/A02 정확성 검사와 A03의 실제 계약 CPU/mock 검증을 제공하며, B~F의 최종 소비자
통합과 총괄 I02의 workspace 인수는 후속 단계다. 동일 SHA·revision·명령·결과로 인계한다.
