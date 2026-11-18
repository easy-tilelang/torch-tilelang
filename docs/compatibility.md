# Compatibility contract

## Compiler baseline

The tested compiler baseline is TileLang
`72d986e1c45ba8e81f7611eb6835c9bece097d60`, with its bundled TVM at
`0215a96cba91eabac903afa4ec1e42c99257e4c3`, built with `USE_ASCEND=ON`.
The default AscendC path uses TileLang's existing interfaces directly. The
verification environment and limits are recorded in [verification](verification.md).

A later AscendC NPU acceptance record uses TileLang
`6c9232103a9cba8177d7ac766f2465f1230f0a19` with the same TVM revision,
Torch 2.12.0, torch_npu 2.12.0.post2 and CANN 9.3 on Ascend950PR_957b.
Its numerical and 30-minute fixed-shape soak checks passed. Performance and PTO
limitations are recorded separately in [verification](verification.md); this
does not establish a general release-version range.

The package declares `torch` and `tilelang` dependencies. This experimental
snapshot has no verified release-version range yet; use the tested revision
when reproducing its compiler checks. A generic TileLang wheel may omit Ascend
or Relax support.

Use the TVM supplied by that TileLang build. A second `apache-tvm` distribution
can select incompatible IR and native FFI implementations. Required compiler
interfaces are:

- TVM `TorchFXImporter`, `relax`, `tirx` and `s_tir`;
- TileLang `backend.create_backend_context`, `compile` and `lower`;
- the Ascend target, native AscendC codegen and `tvm_ffi` execution backend.

Publish new dependency ranges only after the full test matrix has run against
them.

`compat/tilelang.py` owns target resolution and public compiler calls.
`compat/profiler.py` translates the typed missing-measurement exception.
The graph compiler uses its own namespace and has no dependency on TileLang's
source-tree graph modules or test helpers.

## Device and toolchain environment

Keep Torch and torch_npu as a tested pair. Record CANN, Bisheng, PTODSL and
ptoas versions alongside that pair. The original PR reports CANN 9.3 and a
2026-10-20 Bisheng build on Ascend950PR_957b; collect its Torch and torch_npu
versions from that machine before reproducing the combination.

The default target retains the original PR's `dav-3510` architecture. Use
explicit target options for the device being validated. Default execution is
AscendC/TVM-FFI; a PTO target selects the existing PTO variant and adapter.
For graph compilation, PTO with `execution_backend="auto"` is explicitly
resolved to the PTO adapter even when TileLang's own automatic preference is
TVM-FFI. An explicit PTO target with `execution_backend="tvm_ffi"` is rejected.
This graph-specific policy does not change TileLang's backend registry or
other compilation clients. Missing PTO dependencies still produce an error.

## Interfaces

Public graph entry points are `GraphConfig`, `make_backend` and the
`torch_tilelang_ascend` Dynamo name. `fx_to_relax`, `build_runner` and schedule
functions remain experimental interfaces.

Kernels receive input buffers followed by caller-allocated output buffers.
TorchGraphRunner owns GM allocation and last-use release. TileLang owns
compilation, artifact caching, stream-aware launch and execution errors.
Graph grouping and schedule selection belong to this package; the existing
Ascend pass pipeline and codegen belong to TileLang.

Timing defaults to device events. Optional msprof timing is available only
when TileLang exposes `tilelang.profiler.errors.ProfileUnavailableError`.
Otherwise it reports unavailable measurement and rejects the fusion candidate;
the default compilation path is unaffected. Generic RuntimeError and device
execution failures are never translated into missing measurements.

PTO is an optional backend with its own PTODSL, ptoas and Bisheng requirements.
The original compiler baseline has PTO compilation failures for wide rectangular
GEMM epilogues (`M=32, N=64`) and selects whose predicates become Python bools.
The fixes are available in
[`erhsh/tilelang` at `de9c5126`](https://github.com/erhsh/tilelang/commit/de9c512609dffe7956b2e0a1a395fed6c1e7e923),
based on upstream `d2dbf20c`. These changes require rebuilding TileLang's native
compiler as well as using its matching Python package. The tested combination
and validation limits are recorded in
[verification](verification.md#pto-compiler-fixes-2026-11-14); they do not establish
a general PTO release-version range.

With this fixed compiler, the unified PTO source/runtime/cache gate and a
30-minute strict fixed-shape soak passed on the recorded environment. The
subsequent timing batch was blocked by other device processes; end-to-end
performance acceptance remains open. See the
[unified PTO acceptance record](verification.md#unified-pto-acceptance-2026-11-16).

CANN builds sharing the 9.3.0 version label are not interchangeable acceptance
baselines. With the same Torch pair, build `V100R001C13B046` produced a pure
Torch device-side `signbit` discrepancy; the `V100R001C13B023` control passed.
Use the recorded build identifier and toolchain, not the version label alone.
The [toolchain comparison](verification.md#card-5-recovery-and-toolchain-comparison-2026-11-18)
includes the minimal probe and the completed B023 performance batch, whose
timing remained inconclusive.
