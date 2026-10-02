# F 검증 자료 보존 — 2026-10-03

사용자가 원격 보존을 지시한 F 소유 자료다. 신규 F 기능은 변경하지 않았다.
원본 JSON 52개, package text metadata, 실패 2개를 포함한 과거 실행 기록 22개와
별도로 실행한 보존용 검사 13개를 포함한다. 입력은 자체 MIT 합성 fixture이며
검사·학습·설치는 실제 CPU 프로세스에서 실행했다. 실제 Maia/GPU/대국 자료는 없다.

자료 전체 목록·byte SHA-256·크기·출처·소스 연결·합성 여부는 [manifest.json](manifest.json),
전체 파일 checksum은 [SHA256SUMS](SHA256SUMS)에 있다. manifest 자체도 checksum에
포함하며 checksum 파일 자신은 self-reference를 피하기 위해 제외한다.

## 구성

| 경로 | 보존 내용 |
|---|---|
| `original/f01/` | 당시 split plan·감사 보고서·F01 설치 text metadata |
| `original/f02/` | 당시 recipe·모든 번호 checkpoint·최종 재개 지점·receipt·export·검증·summary·설치 metadata; preliminary WIP도 유지 |
| `inputs/f01/`, `inputs/f02/` | 원래 Git commit의 fixture 실행 입력을 byte 그대로 복제, blob OID 포함 |
| `logs/historical/` | 이 대화의 저장된 commandExecution에서 복구한 완전한 combined output·명령·exit code·실패 기록 |
| `commands/historical.json` | 과거 실행 명령 원장; 당시 exact source SHA가 기록되지 않은 항목은 null |
| `logs/recheck/`, `commands/recheck.json` | 보존용 새 실행의 stdout/stderr와 source SHA·시각·시간·exit code·명령 |
| `recheck/runs/`, `recheck/package/` | 새 감사·fixture 연속/중단/재개/frozen·export·설치 검사 결과와 package text metadata |
| `reconstructed/training-data-audit.json` | 당시 저장되지 않았던 감사 본문을 동일 소스로 재구성, 원래 provenance digest와 일치 |
| `derived/original-training-history.tsv` | 원본 receipt.history에서 파생한 TSV; 당시 생성한 raw TSV가 아님 |
| `provenance.json`, `omissions.json` | 소스 byte 대조 근거·미상 WIP·부재/제외/검증 한계 |
| `privacy/` | 절대 경로 마스킹 원장·비밀정보/개인정보/권리 검사 범위와 결과 |

원본 결과 JSON은 byte를 바꾸지 않았다. historical combined logs는 tool에서
`truncated=false`로 제공된 원문이며 stdout/stderr 분리가 없는 capture다. 개인·작업
절대 경로는 논리 변수로 바꿨고, 변경 전/후 byte SHA를 기록한다. 전체 채팅·사용자
신원·다른 역할의 메시지를 복제하지 않았다. 원래 별도 로그 파일이 존재했다는
뜻이 아니며 새 재검사를 당시 실행으로 바꾸어 기록하지 않았다.

archive 내부 `.gitattributes`는 OS별 줄바꿈 변환을 막는다. 원래 package metadata의
CRLF·공백·마지막 빈 줄도 byte 보존 대상이므로 정리하지 않으며, 이 경로의 whitespace
검사 예외를 명시했다. 루트 attributes나 다른 역할의 설정은 변경하지 않았다.

## 소스와 실제 결과

F01 final package는 `387790173754992979902b233af443b5fa2006c1`, F02 검증 package는
`8aa798e6fe3e760be125c5c79da869a1a9c46378`의 소스 bytes와 대조했다. F02 Python
17개 파일은 보존 시작점 `36e619bae238a587e04c0a4bae24624eb5414aec`에서도 동일하다.
이는 사후 소스 일치 증거이며 모든 옛 명령의 실행 당시 HEAD를 증명하지는 않는다.
초기 F01 package와 F02 preliminary-v1은 정확한 Git 소스가 미상임을 유지한다.

보존용 새 검사는 source SHA를 잠그고 105개 regression, F01 split/audit, 같은 Limits의
training audit 재구성, 8 step 연속/3 step 중단+재개, frozen, export import, package
설치·설치 CLI 재개·설치 export 검사를 실행했다. **13개 명령 exit 0**.
새 연속/재개/설치 재개 및 원래 연속 실행의 model·optimizer·sampler·best·history·
provenance가 동일했다. frozen weights는 불변이었다. 원래 training audit digest도
재구성 결과와 일치한다. 실제 시간과 새 receipt/checkpoint digest는 재실행마다 달라진다.
자세한 실제 결과는 [새 검사 요약](recheck/summary.json)에 있다.

## 확인과 재현

archive checksum은 외부 dependency 없이 확인한다.

```bash
python experiments/model-research/verification/2026-10-03-f-evidence-v1/verify_bundle.py
```

아래 명령은 **새 보존용 재검사**를 한다. Python 3.12+, 설치된 pip/setuptools를
사용하고 GPU·네트워크·외부 가중치가 필요하지 않다. source SHA의 library/tests/fixtures/
pyproject와 checkout bytes가 다르면 거부한다. artifact root는 저장소 밖의 새 경로여야 한다.
번호별 timeout은 30초이며 fixture recipe는 8 step/16 samples/10초/출력 1 MiB를 잠근다.

```bash
python experiments/model-research/verification/2026-10-03-f-evidence-v1/reproduce.py \
  --checkout "$PWD" \
  --artifact-root "${ARTIFACT_ROOT:?저장소 밖 root 지정}/f-preservation-replay-v2"
```

이 harness는 새 raw stdout/stderr·입력·결과·명령을 지정한 외부 경로에 기록하며 archive를
덮어쓰지 않는다. 원장 안의 `${CHECKOUT_ROOT}`, `${BUNDLE_ROOT}`, `${ARTIFACT_ROOT}`,
`${PYTHON}`, `${TMPDIR}`는 공개용 논리 경로다. 과거 combined 명령 원장은 읽기 자료이며
직접 실행하는 shell script가 아니다. 재현 시 소스 SHA는 고정해도 실행 시간·wheel ZIP·
새 artifact digest는 달라질 수 있다. 기존 wheel SHA는 과거 pip 로그의 보고값으로만 남고
wheel bytes가 존재하지 않아 재대조할 수 없다.

## 공개 검사·제외·보존

선택한 UTF-8 자료에 대해 credential token/private key/bearer·secret assignment,
email·개인 home 경로·작업 절대 경로·NUL을 검사하고 독립 검토를 수행했다. 원본 설치
metadata 3개의 local URL과 로그/새 metadata의 작업 경로를 마스킹했다. 남은 후보는 0이다.
source/fixture는 자체 MIT이며 외부 데이터·권리 미확인 가중치를 포함하지 않는다.
secret 파일은 열지 않았다. 이 검사는 heuristic+review이며 임의의 개인정보 부재를
수학적으로 증명하는 검사는 아니다.

build cache·복제 source/installed Python·bytecode·wheel/tar·실행 바이너리는 제외했다.
합성 모델의 작은 JSON tensor와 학습 state는 MIT fixture 산출물로 보존한다. 파일당
1 MiB/전체 5 MiB를 보존 상한으로 두었고 실제 합계·최대 크기는 manifest에 기록한다.
상한을 넘는 자료는 바로 Git에 넣지 않고 LFS 또는 checksum-addressed 별도 object
storage/release attachment의 접근·권리·보존 기간을 먼저 보고하도록 한다.

사용자의 이번 지시에 따라 일반적인 raw output 저장소 밖 보관 규약에 작은 검토 자료의
원격 archive 예외를 적용했다. 원래 scratch 자료와 새 검사 raw logs는 삭제하지 않는다.
원격 tree/blob의 실제 bytes와 SHA256SUMS를 대조한 뒤에만 보존 완료로 보고한다.
D/E는 각 소유 경로·작업의 범위이며 이 F archive에는 해당 역할 파일을 섞지 않았다.
