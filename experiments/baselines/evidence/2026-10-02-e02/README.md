# E02 보존 checkpoint

원본 실행 source revision은 `e48ff065a431bc6bdd835d75e46e2ed121661f53`이다. 이 디렉터리는 기존 실행 뒤에 보존한 snapshot이며 실행을 새로 수행한 결과가 아니다. E01 문맥은 `e01-context/`에 분리했고 source revision은 `c2f944f1794b44081905b4019643f6b2978d8feb`이다.

`reports/*-test.log`는 원래 저장된 실제 Cargo test 합본 stdout/stderr이다. CLI별 argv, exit code와 stdout/stderr는 `reports/cli-receipt.json`에 원래 저장된 값이며 `original-command-extract.json`은 그 값의 추출본이다. `runs/`는 full 입력·입력 lock·plan·24개 Event·25개 단계별 JSONL ledger·삭제-prefix 검사용 원장을 그대로 보존한다. 모든 대국 결과와 `n0..n4`는 synthetic declaration이며 실제 Rules 판정이나 runner 대국을 관측하지 않았다. `derived/*.tsv`는 보존 시점에 원본 ledger/receipt에서 생성한 표이다.

환경 절대경로는 `${ARTIFACT_ROOT}`, `${REPO_ROOT}`, `${CARGO}`, `${CARGO_HOME}` 등의 placeholder로 치환했다. 치환 파일의 `INVENTORY.json` 항목은 category/count와 원본·보존본 각각의 SHA-256을 기록한다. 원본은 삭제하지 않았다. 입력 lock/plan/Event/ledger 자체에는 경로 치환이 필요하지 않아 원본과 byte 동일하다. Inventory 자체 SHA는 자기참조를 피하기 위해 포함하지 않는다.

당시 raw Cargo 검사 78개(31+47)의 기록과 입력 manifest를 보존한다. 현재 runner 구현이나 이후 검사를 이 checkpoint의 실행으로 소급하지 않는다. 바이너리, Cargo build cache, 외부 도구 checkout, 모델/weights/data, 개인 비밀은 포함하지 않는다. 참조된 E01 fixture 파일은 해당 source revision의 `experiments/baselines/fixtures/`에 이미 존재한다.

## 재현

`REPRODUCE.md`와 `reproduce-e02.sh`는 원본 실행 이후 작성한 재현 절차이며 아직 실행하지 않은 명령이다. 원래 실제 명령은 receipt/extract를 기준으로 확인한다. 실행하려면 별도 checkout을 당시 source revision에 맞추고, `${CARGO}`와 `${ARTIFACT_ROOT}`를 지정한다. 재현 출력은 새 외부 디렉터리에 기록하며 보존 원본을 덮어쓰지 않는다. 당시 private crate Cargo.lock은 `reports/`에 보존되어 있다.
