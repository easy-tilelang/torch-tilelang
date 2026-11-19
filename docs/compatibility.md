# Compatibility

## Compiler and runtime requirements

The package declares Python 3.10 or later, Torch and TileLang. Build TileLang
with `USE_ASCEND=ON` and use its bundled TVM Python and native libraries.
The frontend uses TVM's Relax FX importer. Scheduling requires `tirx` and
`s_tir`. NPU execution requires a compatible Torch, torch_npu and CANN
installation. PTO execution also requires PTODSL, ptoas and Bisheng.

The graph backend checks the native code generator when it resolves the target.

| Target | Execution adapter | Required TVM global function |
| --- | --- | --- |
| AscendC | `tvm_ffi` | `target.build.tilelang_ascend` |
| PTO | `pto` | `target.build.tilelang_pto_without_compile` |

The optional `msprof` fusion profiler also uses TileLang's
`tilelang.profiler.errors.ProfileUnavailableError`. A build without the
`tilelang.profiler.errors` module can use the default device event timer.
For target and profiler options, see the [Ascend guide](ascend.md).

## Previously tested environments

Earlier verification runs used the following TileLang revisions with bundled
TVM `0215a96cba91eabac903afa4ec1e42c99257e4c3`. Package metadata leaves
the Torch and TileLang versions unpinned.

| Check | TileLang revision |
| --- | --- |
| CPU tests and AscendC source generation | `72d986e1c45ba8e81f7611eb6835c9bece097d60` |
| AscendC device execution | `6c9232103a9cba8177d7ac766f2465f1230f0a19` |
| PTO device execution | [`erhsh/tilelang` at `de9c5126`](https://github.com/erhsh/tilelang/commit/de9c512609dffe7956b2e0a1a395fed6c1e7e923) |

The CPU run used Python 3.14 on Ubuntu under WSL. Its pip constraints are in
[cpu-wsl-py314.txt](../constraints/cpu-wsl-py314.txt).

The device runs used Ascend950PR_957b, Torch `2.12.0+cpu`, torch_npu
`2.12.0.post2` and CANN 9.3.0 build `V100R001C13B023`. The PTO run used
ptoas 0.69 and a compiler with fixes for rectangular GEMM epilogues and scalar
`pto.select` predicates. With the same Torch pair under CANN build
`V100R001C13B046`, device-side fp16 `torch.signbit` disagreed with the CPU
result during signed-zero validation. The original runs are recorded in
`git show 40f1126:docs/verification.md`.

To check your installed compiler and device, follow the
[verification guide](verification.md).
