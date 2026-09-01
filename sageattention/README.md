# Managed SageAttention wheels

This directory builds a companion `comfy-kitchen-sageattention` distribution
from the pinned official SageAttention source. The companion owns the
`sageattention` import namespace while keeping its Torch ABI-sensitive native
extensions out of the stable-ABI `comfy-kitchen` wheel.

The source archive, upstream commit, license, Torch/CUDA line, and packaging
revision are pinned in `source.json`. `prepare_source.py` verifies the archive
before extracting it and applies `packaging.patch`. Published wheels include
the upstream Apache-2.0 license, attribution, and machine-readable provenance.
The build uses the pinned source archive timestamp for deterministic wheel
timestamps. The pinned upstream source has no NOTICE file.

The managed patch includes upstream commit
`e5bf6ee147e00107cb65042787d3a34b2e014c8c`, which restores direct native
extension calls and removes the compile wrappers associated with incorrect
SM90 output reports. The Linux build also omits upstream debug and OpenMP
flags because the extension sources do not use OpenMP.

The first binary line supports Python 3.12, Torch 2.13.0, CUDA 13.0, Linux
x86_64, and Windows x86_64. Its compiled arms cover the upstream 2.2.0 dispatch
set: SM80, SM86, SM89, SM90, and SM120. Triton 3.7.1 is installed as a runtime
dependency; Windows uses `triton-windows==3.7.1.post27`.

Prepare the source without changing the checkout:

```bash
python sageattention/prepare_source.py sageattention/build/source
```

Use `--archive` for an already-downloaded source archive. The GitHub Actions
workflow builds and verifies manylinux x86_64 and win_amd64 wheels, including
the cubin coverage of each native extension. A dedicated
`sageattention-v2.2.0.post1` tag publishes only after both platform jobs pass.
