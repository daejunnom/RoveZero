"""Verify the preserved archive hashes and embedded F JSON receipts offline."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_bytes())
    sealed=0
    for row in manifest['files']:
        logical=PurePosixPath(row['path'])
        assert not logical.is_absolute() and '..' not in logical.parts,row['path']
        data=(root/row['path']).read_bytes()
        assert len(data)==row['size_bytes'],row['path']
        assert sha(data)==row['byte_sha256'],row['path']
        if row.get('embedded_payload_digest') is not None:
            value=json.loads(data)
            canonical=json.dumps({k:v for k,v in value.items() if k!='digest'},sort_keys=True,
                                 ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
            assert sha(canonical)==value['digest']==row['embedded_payload_digest'],row['path']
            sealed+=1
    checked=[]
    for line in (root/'SHA256SUMS').read_text().splitlines():
        checksum,name=line.split('  ',1)
        logical=PurePosixPath(name)
        assert not logical.is_absolute() and '..' not in logical.parts,name
        assert sha((root/name).read_bytes())==checksum,name
        checked.append(name)
    assert len(checked)==len(set(checked)), 'duplicate checksum entries'
    expected={row['path'] for row in manifest['files']} | {'manifest.json'}
    assert set(checked)==expected, 'index/checksum coverage mismatch'
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==expected | {'SHA256SUMS'}, 'unexpected or missing archive file'
    print(json.dumps({'files_verified':len(checked),'embedded_receipts_verified':sealed,
                      'indexed_bytes':manifest['indexed_bytes'],'original_json_files':manifest['original_json_files'],
                      'historical_captures':manifest['historical_command_captures'],
                      'new_recheck_commands':manifest['new_recheck_commands']}))


if __name__=='__main__':
    main()
