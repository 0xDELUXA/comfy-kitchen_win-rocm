"""HIP dispatch and gating logic. Needs no GPU and no compiled extension.

These cover which architectures the backend registers for and which ops it
advertises on each. They are deliberately not gated on registry.is_available("hip"):
on a CPU-only runner the kernel suite in test_hip_wmma.py skips in full, and these
routing rules would otherwise go untested behind a green tick.
"""
import ast
import pathlib
import re
import subprocess
import sys

import pytest
import torch

import comfy_kitchen.scaled_mm_v2 as scaled_mm_module
from comfy_kitchen.backends import hip as hip_backend

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_HIP_CMAKE = _ROOT / "comfy_kitchen" / "backends" / "hip" / "CMakeLists.txt"


def test_non_rocm_runtime_does_not_import_hip_backend():
    """A combined wheel must not load the ROCm runtime in CUDA/CPU processes."""
    if getattr(torch.version, "hip", None):
        pytest.skip("requires a non-ROCm PyTorch runtime")

    code = """
import sys
import comfy_kitchen as ck

assert "comfy_kitchen.backends.hip" not in sys.modules
status = ck.list_backends()["hip"]
assert not status["available"]
assert status["unavailable_reason"] == "PyTorch ROCm/HIP runtime not available"
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_scaled_mm_does_not_probe_hip_on_non_rocm_runtime(monkeypatch):
    """Keep the HIP routing check off NVIDIA's latency-sensitive FP8 path."""
    if getattr(torch.version, "hip", None):
        pytest.skip("requires a non-ROCm PyTorch runtime")

    sentinel = object()

    def unexpected_hip_probe(*args, **kwargs):
        raise AssertionError("HIP was probed without a ROCm PyTorch runtime")

    monkeypatch.setattr(scaled_mm_module, "_hip_fp8_gemm", unexpected_hip_probe)
    monkeypatch.setattr(scaled_mm_module, "has_scaled_mm_v2", lambda: True)
    monkeypatch.setattr(torch.nn.functional, "scaled_mm", lambda *args, **kwargs: sentinel)

    result = scaled_mm_module.scaled_mm_v2(
        object(),
        object(),
        object(),
        object(),
    )
    assert result is sentinel


@pytest.mark.parametrize(
    "arches",
    [
        [],
        ["gfx90a"],   # CDNA: MFMA, not WMMA
        ["gfx1010"],  # RDNA1: neither matrix cores nor the dot-product paths
        ["gfx1201", "gfx90a"],
    ],
)
def test_hip_declines_unsupported_arch(arches):
    """The backend registers for RDNA2/3/4 and nothing else."""
    assert hip_backend._unsupported_arch_reason(arches) is not None


@pytest.mark.parametrize(
    "arches",
    [["gfx1201", "gfx1200"], ["gfx1100"], ["gfx1151"], ["gfx1030"], ["gfx1200", "gfx1030"]],
)
def test_hip_accepts_rdna2_through_rdna4(arches):
    assert hip_backend._unsupported_arch_reason(arches) is None


def test_hip_declines_when_an_arch_cannot_be_read():
    """A device whose architecture is unknown cannot be shown to be supported."""
    assert hip_backend._unsupported_arch_reason([None]) is not None
    assert hip_backend._unsupported_arch_reason(["gfx1200", None]) is not None


@pytest.mark.parametrize(
    ("arches", "expected"),
    [
        (["gfx1200"], True),
        (["gfx1201", "gfx1100"], True),
        (["gfx1151"], True),
        (["gfx1030"], False),             # RDNA2 has no matrix cores
        (["gfx1200", "gfx1030"], False),  # kernels launch on the tensor's own device
        ([None], False),
    ],
)
def test_hip_wmma_capability(arches, expected):
    """Only an all-matrix-core process may advertise the GEMMs."""
    assert hip_backend._has_wmma(arches) is expected


def test_hip_drops_gemms_without_matrix_cores():
    """RDNA2 keeps the elementwise kernels and hands the GEMMs back to triton/eager."""
    with_wmma = hip_backend._build_constraints(has_wmma=True)
    without = hip_backend._build_constraints(has_wmma=False)

    # Every WMMA-only op must be advertised with matrix cores; an intersection
    # would pass while any single one was missing from the constraints.
    assert set(with_wmma) >= hip_backend._WMMA_ONLY_OPS
    assert not (hip_backend._WMMA_ONLY_OPS & set(without))
    # The elementwise kernels need no matrix cores and must survive.
    for op in ("apply_rope", "apply_rope_", "rms_rope", "rms_rope_split_half1_", "adaln",
               "rms_adaln", "quantize_per_tensor_fp8", "gemv_awq_w4a16"):
        assert op in without


def test_hip_advertises_every_inplace_rope_entry():
    """A missing entry routes the in-place call to eager while the functional one
    stays on HIP: silently half the coverage rather than a failure."""
    constraints = hip_backend._build_constraints(has_wmma=True)
    for functional in ("apply_rope", "apply_rope1", "apply_rope_split_half",
                       "apply_rope_split_half1", "rms_rope", "rms_rope1",
                       "rms_rope_split_half", "rms_rope_split_half1"):
        assert constraints[f"{functional}_"] is constraints[functional]


def _setup_py_default_archs() -> list[str]:
    """DEFAULT_HIP_ARCHS, read without importing setup.py (which would run setup())."""
    tree = ast.parse((_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_HIP_ARCHS" for t in node.targets
        ):
            return ast.literal_eval(node.value).split(";")
    raise AssertionError("DEFAULT_HIP_ARCHS not found in setup.py")


def _setup_namespace() -> dict:
    """Load setup.py definitions without executing its final setuptools.setup()."""
    path = _ROOT / "setup.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "extensions"
            for target in node.targets
        ):
            break
        body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    namespace = {"__file__": str(path), "__name__": "comfy_kitchen_setup_test"}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_setup_keeps_cuda_build_cuda_only_by_default():
    namespace = _setup_namespace()
    cuda_extension = object()

    namespace["setup_cuda_extension"] = lambda: cuda_extension
    namespace["setup_hip_extension"] = lambda: pytest.fail(
        "an incidental ROCm compiler must not add HIP to a CUDA source build"
    )

    assert namespace["get_extensions"]() == [cuda_extension]


def test_setup_builds_both_backends_only_when_hip_is_requested():
    namespace = _setup_namespace()
    cuda_extension = object()
    hip_extension = object()

    namespace["BUILD_HIP"] = True
    namespace["setup_cuda_extension"] = lambda: cuda_extension
    namespace["setup_hip_extension"] = lambda: hip_extension

    assert namespace["get_extensions"]() == [cuda_extension, hip_extension]


def test_setup_refuses_hip_only_fallback_under_cuda_pytorch():
    namespace = _setup_namespace()
    missing_cuda = namespace["CudaToolkitNotFoundError"]

    def raise_missing_cuda():
        raise missing_cuda("nvcc missing")

    namespace["setup_cuda_extension"] = raise_missing_cuda
    namespace["get_rocm_path"] = lambda: ("/opt/rocm", object())
    namespace["get_torch_gpu_runtime"] = lambda: "cuda"
    namespace["setup_hip_extension"] = lambda: pytest.fail(
        "CUDA PyTorch must not silently receive a HIP-only native build"
    )

    with pytest.raises(missing_cuda, match="refusing to replace"):
        namespace["get_extensions"]()


def test_no_cuda_means_python_only_unless_hip_is_explicit():
    namespace = _setup_namespace()
    namespace["BUILD_NO_CUDA"] = True
    namespace["setup_cuda_extension"] = lambda: pytest.fail("CUDA was not disabled")
    namespace["setup_hip_extension"] = lambda: pytest.fail("HIP was not requested")

    assert namespace["get_extensions"]() == []


def test_rocm_only_build_still_auto_selects_hip():
    namespace = _setup_namespace()
    missing_cuda = namespace["CudaToolkitNotFoundError"]
    hip_extension = object()

    def raise_missing_cuda():
        raise missing_cuda("nvcc missing")

    namespace["setup_cuda_extension"] = raise_missing_cuda
    namespace["get_rocm_path"] = lambda: ("/opt/rocm", object())
    namespace["get_torch_gpu_runtime"] = lambda: "hip"
    namespace["setup_hip_extension"] = lambda: hip_extension

    assert namespace["get_extensions"]() == [hip_extension]


def _cmake_default_archs() -> list[str]:
    text = _HIP_CMAKE.read_text(encoding="utf-8")
    block = re.search(r"set\(COMFY_HIP_ARCHS\s*(.*?)\)", text, re.DOTALL)
    assert block, "COMFY_HIP_ARCHS default not found in CMakeLists.txt"
    return re.findall(r'"(gfx\w+)"', block.group(1))


def test_hip_default_archs_match_the_cmake_default():
    """setup.py always passes -DCOMFY_HIP_ARCHS, so a drifted CMake default shows
    up only in a direct cmake run, silently missing whichever target was added."""
    assert _setup_py_default_archs() == _cmake_default_archs()


def test_hip_kernels_are_independent_of_the_python_extension_target():
    """Keep expensive HIP objects reusable across the CPython wheel matrix."""
    text = _HIP_CMAKE.read_text(encoding="utf-8")

    assert "add_library(comfy_kitchen_hip_kernels OBJECT ${HIP_SOURCES})" in text
    assert "target_sources(_C PRIVATE $<TARGET_OBJECTS:comfy_kitchen_hip_kernels>)" in text

    module_calls = re.findall(r"nanobind_add_module\((.*?)\)", text, re.DOTALL)
    assert module_calls
    assert all("${HIP_SOURCES}" not in call for call in module_calls)


def test_combined_wheel_cache_keys_include_hip_sources():
    """HIP-only changes must produce a cache key that Actions can save."""
    workflow = (_ROOT / ".github" / "workflows" / "build-wheels.yml").read_text(
        encoding="utf-8"
    )
    combined_cache_keys = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("key: ccache-")
        and ("linux-x86_64" in line or "windows-x86_64" in line)
    ]

    assert len(combined_cache_keys) == 2
    assert all("'comfy_kitchen/backends/hip/**'" in key for key in combined_cache_keys)
