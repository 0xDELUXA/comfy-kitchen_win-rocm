from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "source.json"
PATCH = ROOT / "packaging.patch"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "comfy-kitchen-build"})
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def _load_manifest() -> dict[str, Any]:
    data: object = json.loads(MANIFEST.read_text())
    if not isinstance(data, dict):
        raise ValueError("source.json must contain an object")
    return data


def prepare_source(destination: Path, archive: Path | None = None) -> None:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    manifest = _load_manifest()
    upstream = manifest["upstream"]
    commit = upstream["commit"]
    expected_root = f"SageAttention-{commit}"
    source_date_epoch = manifest["build"]["source_date_epoch"]
    environment_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if environment_epoch is not None and int(environment_epoch) != source_date_epoch:
        raise ValueError(f"SOURCE_DATE_EPOCH is {environment_epoch}, expected {source_date_epoch}")

    with tempfile.TemporaryDirectory(prefix="comfy-kitchen-sageattention-") as temporary:
        temporary_path = Path(temporary)
        source_archive = archive if archive is not None else temporary_path / "source.tar.gz"
        if archive is None:
            _download(upstream["archive_url"], source_archive)

        actual_size = source_archive.stat().st_size
        if actual_size != upstream["archive_size"]:
            raise ValueError(
                f"source archive size is {actual_size}, expected {upstream['archive_size']}"
            )
        actual_digest = _sha256(source_archive)
        if actual_digest != upstream["archive_sha256"]:
            raise ValueError(
                f"source archive sha256 is {actual_digest}, expected {upstream['archive_sha256']}"
            )

        extracted = temporary_path / "extracted"
        extracted.mkdir()
        with tarfile.open(source_archive, "r:gz") as bundle:
            members = bundle.getmembers()
            roots = {Path(member.name).parts[0] for member in members if member.name}
            if roots != {expected_root}:
                raise ValueError(
                    f"source archive root is {sorted(roots)}, expected {expected_root}"
                )
            root_members = [
                member
                for member in members
                if Path(member.name).parts == (expected_root,) and member.isdir()
            ]
            if len(root_members) != 1 or root_members[0].mtime != source_date_epoch:
                actual_mtimes = [member.mtime for member in root_members]
                raise ValueError(
                    f"source archive root mtime is {actual_mtimes}, expected {source_date_epoch}"
                )
            bundle.extractall(extracted, filter="data")

        source = extracted / expected_root
        subprocess.run(["git", "apply", "--unidiff-zero", str(PATCH)], cwd=source, check=True)
        for relative_path in manifest["backport"]["removed_files"]:
            backported_removal = source / relative_path
            if not backported_removal.is_file():
                raise FileNotFoundError(f"backport removal is missing: {relative_path}")
            backported_removal.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify, extract, and patch the pinned SageAttention source"
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    prepare_source(args.destination.resolve(), args.archive.resolve() if args.archive else None)


if __name__ == "__main__":
    main()
