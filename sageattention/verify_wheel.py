from __future__ import annotations

import argparse
import email.parser
import json
import zipfile
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "source.json").read_text())
PLATFORM_TAGS = {
    "linux_x86_64": "cp312-cp312-linux_x86_64",
    "manylinux_x86_64": "cp312-cp312-manylinux_2_28_x86_64",
    "win_amd64": "cp312-cp312-win_amd64",
}


def _single(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one *{suffix} member, found {matches}")
    return matches[0]


def verify_wheel(path: Path, platform: str) -> None:
    distribution = MANIFEST["distribution"]
    upstream = MANIFEST["upstream"]
    backport = MANIFEST["backport"]
    build = MANIFEST["build"]
    normalized = distribution["name"].replace("-", "_")
    dist_info = f"{normalized}-{distribution['version']}.dist-info/"

    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata_name = _single(names, ".dist-info/METADATA")
        wheel_name = _single(names, ".dist-info/WHEEL")
        if not metadata_name.startswith(dist_info) or not wheel_name.startswith(dist_info):
            raise ValueError("wheel dist-info directory does not match the managed distribution")

        metadata = email.parser.BytesParser().parsebytes(wheel.read(metadata_name))
        if metadata["Name"] != distribution["name"]:
            raise ValueError(f"unexpected distribution name: {metadata['Name']}")
        if metadata["Version"] != distribution["version"]:
            raise ValueError(f"unexpected distribution version: {metadata['Version']}")
        if metadata["Requires-Python"] != ">=3.12,<3.13":
            raise ValueError(f"unexpected Python requirement: {metadata['Requires-Python']}")

        parsed_requirements = [Requirement(raw) for raw in metadata.get_all("Requires-Dist", [])]
        requirements = {requirement.name: requirement for requirement in parsed_requirements}
        if set(requirements) != {"torch", "triton", "triton-windows"}:
            raise ValueError(f"unexpected runtime requirements: {requirements}")
        expected_versions = {
            "torch": build["torch"],
            "triton": build["triton_linux"],
            "triton-windows": build["triton_windows"],
        }
        for name, version in expected_versions.items():
            if str(requirements[name].specifier) != f"=={version}":
                raise ValueError(f"unexpected {name} requirement: {requirements[name]}")
        expected_markers = {
            "torch": None,
            "triton": 'sys_platform != "win32"',
            "triton-windows": 'sys_platform == "win32"',
        }
        for name, marker in expected_markers.items():
            actual = str(requirements[name].marker) if requirements[name].marker else None
            if actual != marker:
                raise ValueError(f"unexpected {name} environment marker: {actual}")

        wheel_metadata = wheel.read(wheel_name).decode()
        expected_tag = f"Tag: {PLATFORM_TAGS[platform]}"
        if expected_tag not in wheel_metadata:
            raise ValueError(f"wheel is missing {expected_tag!r}:\n{wheel_metadata}")

        extension_suffix = ".pyd" if platform == "win_amd64" else ".so"
        for stem in ("_fused", "_qattn_sm80", "_qattn_sm89", "_qattn_sm90"):
            if not any(
                name.startswith(f"sageattention/{stem}.") and name.endswith(extension_suffix)
                for name in names
            ):
                raise ValueError(f"wheel is missing the {stem} native extension")

        provenance_name = "sageattention/_comfy_kitchen_provenance.json"
        provenance = json.loads(wheel.read(provenance_name))
        expected_provenance = {
            "distribution": distribution["name"],
            "distribution_version": distribution["version"],
            "upstream_commit": upstream["commit"],
            "upstream_source_sha256": upstream["archive_sha256"],
            "upstream_tag": upstream["tag"],
            "upstream_version": upstream["version"],
            "backport_commit": backport["commit"],
            "python": build["python"],
            "torch": build["torch"],
            "cuda": build["cuda"],
            "build_tools": build["build_tools"],
            "source_date_epoch": build["source_date_epoch"],
        }
        if provenance != expected_provenance:
            raise ValueError(f"wheel provenance does not match source.json: {provenance}")

        removed_modules = set(backport["removed_files"])
        unexpected_modules = [name for name in names if name in removed_modules]
        if unexpected_modules:
            raise ValueError(f"wheel contains removed compile wrappers: {unexpected_modules}")

        license_names = [name for name in names if name.startswith(dist_info)]
        if not any(name.endswith("/LICENSE") for name in license_names):
            raise ValueError("wheel is missing the upstream LICENSE")
        if not any(name.endswith("/NOTICE") for name in license_names):
            raise ValueError("wheel is missing the managed distribution NOTICE")

        forbidden_suffixes = (".cu", ".cuh", ".cpp", ".hpp")
        forbidden = [name for name in names if name.endswith(forbidden_suffixes)]
        if forbidden:
            raise ValueError(f"binary wheel contains build sources: {forbidden}")

    print(f"verified {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a managed SageAttention wheel")
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--platform", choices=tuple(PLATFORM_TAGS), required=True)
    args = parser.parse_args()
    verify_wheel(args.wheel, args.platform)


if __name__ == "__main__":
    main()
