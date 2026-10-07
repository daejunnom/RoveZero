"""Small cache identity/lifetime fixtures; no native code or GPU."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared_runtime_cache import prepare,validate,SCHEMA

@unittest.skipUnless(os.name=="posix","Linux no-follow cache boundary")
class SharedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="rz-shared-cache-")
        self.base=Path(self.temp.name);self.root=self.base/("cuda-"+"a"*64)
        self.root.mkdir()
        self.files=[]
        for index in range(19):
            name=f"libfixture{index}.so.1";data=f"authored-runtime-identity-{index}".encode()
            path=self.root/name;path.write_bytes(data);path.chmod(0o400)
            self.files.append(dict(filename=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        self.root.chmod(0o500)
        self.spec=dict(schema=SCHEMA,cache_directory=str(self.root),canonical_sha256="a"*64,files=self.files)
    def tearDown(self):
        self.root.chmod(0o700)
        self.temp.cleanup()
    def test_read_hash_preserves_files_and_never_loads_native_code(self):
        paths=[self.root/f["filename"] for f in self.files]
        identities=[(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.read_bytes()) for p in paths]
        report=prepare(self.spec,self.base/"report.json")
        self.assertEqual(report["status"],"passed")
        self.assertFalse(report["native_code_loaded"]);self.assertFalse(report["GPU_used"])
        self.assertFalse(report["files_modified"])
        self.assertEqual(identities,[(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.read_bytes()) for p in paths])
    def test_hash_mismatch_rejects_without_success_report(self):
        self.files[0]["sha256"]="b"*64
        with self.assertRaises(ValueError):prepare(self.spec,self.base/"report.json")
        self.assertFalse((self.base/"report.json").exists())
    def test_extra_member_or_symlink_is_not_read(self):
        self.root.chmod(0o700)
        (self.root/"unexpected.fixture").write_bytes(b"not a declared runtime file")
        self.root.chmod(0o500)
        with self.assertRaises(ValueError):validate(self.spec)
        self.root.chmod(0o700);(self.root/"unexpected.fixture").unlink()
        leaf=self.root/self.files[0]["filename"];leaf.unlink();leaf.symlink_to(self.root/self.files[1]["filename"])
        self.root.chmod(0o500)
        with self.assertRaises(OSError):prepare(self.spec,self.base/"report.json")
        self.assertFalse((self.base/"report.json").exists())
    def test_timeout_or_writable_cache_never_reports_prepared(self):
        with self.assertRaises(TimeoutError):prepare(self.spec,self.base/"report.json",wall=0)
        self.root.chmod(0o700)
        with self.assertRaises(ValueError):validate(self.spec)
        self.assertFalse((self.base/"report.json").exists())

if __name__=="__main__":unittest.main()

