# Local verification

Verified on 2026-11-10 in Ubuntu 26.04 under WSL, using Python 3.14.4 and
Torch 2.15.0+cpu. Dependency versions are recorded in
`constraints/cpu-wsl-py314.txt`.

## Native compiler build

Built TileLang at `72d986e1c45ba8e81f7611eb6835c9bece097d60` with its
unmodified compiler/runtime sources, paired with TVM
`0215a96cba91eabac903afa4ec1e42c99257e4c3`. The build used Release, Ascend
enabled, Ascend runtime stubs, CUDA/ROCm/LLVM disabled, and the PyPI Z3
installation. Native libraries were rebuilt for this baseline. This dependency
contains no source-tree graph packages.

## Installed external package

Built the Python wheel and installed it into a separate virtual environment.
Ran `scripts/check_installed.py` from outside the repository. It verified:

- `torch_tilelang` resolves to the wheel's site-packages directory;
- a fresh process finds `torch_tilelang_ascend` before importing the package;
- a CPU inference graph reaches the backend and correctly uses its Torch fallback;
- the dependency has no `tilelang.graph` or `tilelang.ascend.graph`.

The AscendC selection, using that installed wheel and fresh native build:

```text
python -m pytest tests -q -m "not pto" --tb=short -ra
146 passed, 46 skipped, 70 deselected
```

The passes include real FX/Relax conversion, injected CPU execution, schedule
legality, AscendC source generation, dtype/alias/layout regressions, profiler
error handling, and fresh-process package discovery. Of the skips, 45 need real
NPU execution and one needs TileLang's optional typed msprof-error interface.

## Optional PTO selection

The full suite was also run against the same compiler baseline:

```text
python -m pytest tests -q --tb=short -ra
1 failed, 168 passed, 93 skipped
```

The failure is
`test_pto_rectangular_gemm_epilogue_uses_nz_source_stride[wide]`: TileLang's
PTO codegen rejects the NZ source stride for the `M=32, N=64` epilogue.
The regression remains enabled under the `pto` marker. It is not an AscendC
failure. This baseline is not claimed as a working PTO combination.

## Artifacts and remaining device validation

The repository passes `git diff --check`; Python code passes ruff's configured
checks. Wheel and source-distribution builds succeeded. Both archives were
checked for unwanted compiler-fix payloads and contain only this project's
code, tests, examples, documentation and packaging metadata.

This verification does not establish NPU numerical performance, stream
execution, fresh-process PTO device-cache restoration, or PTODSL assembly.
Run the AscendC hardware tests with a matching Torch/torch_npu pair and CANN
compiler environment before claiming device compatibility. PTO additionally
needs PTODSL, ptoas and Bisheng for its assembly and device-cache checks.
