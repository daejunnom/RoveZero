"""Install the pinned official universal release outside Git; never build or run it."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

VERSION = "sf_19"
SOURCE = "edb0d9db6731067ec50ce619ff372b463bc4dd5d"
ARCHIVE = "stockfish-linux-x86-64-universal.tar.gz"
ARCHIVE_BYTES = 81_388_977
ARCHIVE_SHA = "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"
URL = f"https://github.com/official-stockfish/Stockfish/releases/download/{VERSION}/{ARCHIVE}"
CAP = 1024**3

def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()

def outside_git(root):
    if not root.is_absolute():
        raise ValueError("absolute output root required")
    for part in (root, *root.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("linked installation roots are unsupported")
        if (part / ".git").exists():
            raise ValueError("external GPL binaries must be stored outside Git")
    return root

def install(root):
    root = outside_git(root)
    receipt_path = root / "stockfish19-install.json"
    if receipt_path.exists():
        if receipt_path.stat().st_size > 65536:
            raise ValueError("install receipt exceeds bound")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        relative=receipt["binary_relative_path"]
        if not relative.startswith("distribution/stockfish/"):
            raise ValueError("invalid cached binary location")
        binary = root / relative
        if receipt["archive_sha256"] != ARCHIVE_SHA or not binary.resolve().is_relative_to(root.resolve()) or binary.is_symlink() or sha(binary) != receipt["binary_sha256"]:
            raise ValueError("existing installation identity differs; no overwrite/retry")
        return receipt
    root.mkdir(parents=True, exist_ok=False)
    archive = root / ARCHIVE
    # Exact release metadata prevents an automatic latest-version substitution.
    request = urllib.request.Request(f"https://api.github.com/repos/official-stockfish/Stockfish/releases/tags/{VERSION}", headers={"User-Agent": "RoveZero-stockfish19-installer"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(1024**2 + 1)
    if len(body) > 1024**2:
        raise ValueError("release metadata exceeds bound")
    release = json.loads(body)
    assets = [a for a in release["assets"] if a["name"] == ARCHIVE]
    if len(assets) != 1 or assets[0]["size"] != ARCHIVE_BYTES or assets[0]["digest"] != "sha256:" + ARCHIVE_SHA or assets[0]["browser_download_url"] != URL:
        raise ValueError("official release pin differs")
    total = 0
    with urllib.request.urlopen(URL, timeout=60) as response, archive.open("xb") as output:
        while block := response.read(1024**2):
            total += len(block)
            if total > ARCHIVE_BYTES:
                raise ValueError("archive exceeds declared size")
            output.write(block)
    if total != ARCHIVE_BYTES or sha(archive) != ARCHIVE_SHA:
        raise ValueError("official archive checksum/size mismatch; partial files preserved")
    extracted = root / "distribution"
    extracted.mkdir()
    members_count = 0
    expanded = 0
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            members_count += 1
            name = PurePosixPath(member.name)
            if members_count > 128 or name.is_absolute() or ".." in name.parts or "\\" in member.name or len(member.name) > 512 or not (member.isdir() or member.isfile()):
                raise ValueError("unsupported archive path/kind/count")
            destination = extracted.joinpath(*name.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            expanded += member.size
            if expanded > CAP or member.size > CAP:
                raise ValueError("distribution exceeds 1GiB install bound")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with package.extractfile(member) as stream, destination.open("xb") as output:
                shutil.copyfileobj(stream, output, 1024**2)
            if destination.stat().st_size!=member.size:
                raise ValueError("truncated distribution member")
            destination.chmod(0o700 if member.mode & 0o111 else 0o600)
    def is_elf(path):
        with path.open("rb") as stream:
            return stream.read(4)==b"\x7fELF"
    binaries = [p for p in extracted.rglob("*") if p.is_file() and p.name.startswith("stockfish") and is_elf(p)]
    if len(binaries) != 1:
        raise ValueError("expected one universal ELF binary")
    binary = binaries[0]
    receipt = dict(schema="rovezero.external-tool-install.v1", version="19", tag=VERSION,
                   source_commit=SOURCE, source_url="https://github.com/official-stockfish/Stockfish", license="GPL-3.0-only",
                   archive_url=URL, archive_sha256=ARCHIVE_SHA, archive_bytes=ARCHIVE_BYTES,
                   binary_relative_path=binary.relative_to(root).as_posix(), binary_sha256=sha(binary), binary_bytes=binary.stat().st_size,
                   expanded_bytes=expanded, selected_isa="unknown", executed=False)
    with receipt_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.output_root), sort_keys=True))

if __name__ == "__main__":
    main()
