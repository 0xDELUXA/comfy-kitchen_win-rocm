from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "source.json").read_text())
EXPECTED_ARCHITECTURES = {
    stem: set(architectures)
    for stem, architectures in MANIFEST["build"]["extension_architectures"].items()
}


def _package_directory() -> Path:
    spec = importlib.util.find_spec("sageattention")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("installed sageattention package was not found")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(f"expected one sageattention package directory, found {locations}")
    return Path(locations[0])


def verify_architectures(cuobjdump: Path) -> None:
    package = _package_directory()
    for stem, expected in EXPECTED_ARCHITECTURES.items():
        extensions = [
            path for path in package.glob(f"{stem}.*") if path.suffix.lower() in {".pyd", ".so"}
        ]
        if len(extensions) != 1:
            raise RuntimeError(f"expected one {stem} native extension, found {extensions}")
        result = subprocess.run(
            [str(cuobjdump), "--list-elf", str(extensions[0])],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = {int(value) for value in re.findall(r"\.sm_(\d+)(?:a)?\.cubin", result.stdout)}
        if actual != expected:
            raise RuntimeError(
                f"{stem} has architectures {sorted(actual)}, expected {sorted(expected)}"
            )
        print(f"verified {stem}: {', '.join(f'SM{value}' for value in sorted(actual))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify installed SageAttention cubin coverage")
    parser.add_argument("cuobjdump", type=Path)
    args = parser.parse_args()
    verify_architectures(args.cuobjdump.resolve())


if __name__ == "__main__":
    main()
