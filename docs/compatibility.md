# Compatibility

## Compiler and runtime

Build TileLang with `USE_ASCEND=ON` and use its bundled TVM for both the Python
interfaces and the native libraries. The frontend uses TVM's Relax FX importer,
and the Ascend schedules use `tirx` and `s_tir`. The backend uses
TileLang's `create_backend_context`, `compile` and
`lower` interfaces. The package declares `torch` and `tilelang` dependencies
without a verified release-version range.

These source revisions provide reproducible starting points. All three use the
bundled TVM revision `0215a96cba91eabac903afa4ec1e42c99257e4c3`.

| Use | TileLang revision |
| --- | --- |
| CPU-side compiler checks | `72d986e1c45ba8e81f7611eb6835c9bece097d60` |
| AscendC device execution | `6c9232103a9cba8177d7ac766f2465f1230f0a19` |
| PTO device execution | [`erhsh/tilelang` at `de9c5126`](https://github.com/erhsh/tilelang/commit/de9c512609dffe7956b2e0a1a395fed6c1e7e923) |

The PTO revision includes fixes for rectangular GEMM epilogues and scalar
`pto.select` predicates. Rebuild TileLang's native compiler and install the
matching Python package to pick up both fixes.

For Ascend950PR_957b, a reference device configuration is Torch 2.12.0,
`torch_npu` 2.12.0.post2 and CANN 9.3.0 build `V100R001C13B023`. PTO also
uses PTODSL, ptoas 0.69 and the matching Bisheng compiler. With the same Torch
pair, device-side fp16 `torch.signbit` can disagree with CPU in the CANN
`V100R001C13B046` environment. This affects the signed-zero check during graph
validation. Include the CANN build identifier when reporting this issue.

## Backend selection

Use the `tilelang` Dynamo entry point or configure a backend with
`GraphConfig` and `make_backend`. The lower-level `fx_to_relax`, `build_runner`
and schedule functions are experimental.

Set `options={"target": "ascendc"}` or `options={"target": "pto"}` to override
`TILELANG_DEFAULT_TARGET` for a graph. With neither set, the target is
`{"kind": "ascend", "mcpu": "dav-3510"}`. Set the target architecture explicitly
for a different device. The graph backend checks for the selected native code
generator before partitioning the graph.

| Target variant | Adapter selected by `execution_backend="auto"` | Required native code generator |
| --- | --- | --- |
| AscendC | `tvm_ffi` | `target.build.tilelang_ascend` |
| PTO | `pto` | `target.build.tilelang_pto_without_compile` |

An explicit PTO target requires `execution_backend="pto"` or `"auto"`. The
adapter receives Torch input buffers followed by Torch-allocated output buffers.
See the [backend adapter](../src/torch_tilelang/compat/tilelang.py).

Fusion profiling defaults to device events. Selecting `msprof` additionally
requires TileLang's `tilelang.profiler.errors.ProfileUnavailableError`
interface. If it is missing, the selector keeps the unfused baseline.

Use the [verification guide](verification.md) to check a compiler or device
configuration.
