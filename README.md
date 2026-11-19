# torch-tilelang

An experimental `torch.compile` backend for inference on Ascend NPUs.
It converts supported FX partitions through Relax and TIR into TileLang kernels.
Torch allocates the kernel outputs. TileLang compiles, caches and launches the
kernels through the AscendC or PTO backend.

## Installation

Use Python 3.10 or later. Install an Ascend-enabled TileLang build with its
bundled TVM/Relax. To run on an NPU, install compatible versions of Torch,
torch_npu and CANN. Then install this checkout into that environment.
[Compiler and runtime requirements](docs/compatibility.md)

```bash
python -m pip install --no-deps -e .
```

## Quick start

Dynamo discovers the installed `tilelang` backend by name. This example checks
a compiled elementwise expression against Torch on one NPU.

```python
import torch


def model(x, bias):
    return torch.relu(x + bias) * bias


x = torch.randn(7, 65, device="npu")
bias = torch.randn(65, device="npu")
compiled = torch.compile(model, backend="tilelang", fullgraph=True, options={"strict": True})
with torch.no_grad():
    torch.testing.assert_close(compiled(x, bias), model(x, bias))
```

An explicit `options={"target": "ascendc"}` or `options={"target": "pto"}`
takes precedence over `TILELANG_DEFAULT_TARGET`. With neither set, the backend
uses AscendC for `dav-3510`. The name `torch_tilelang_ascend` is also registered.

Compiled partitions accept contiguous fp16, bf16 or fp32 tensors with positive
static dimensions and zero storage offset on one NPU. They support elementwise
expressions, last-axis reductions and aligned 2-D fp16/bf16 GEMM. By default,
a partition that fails to compile or validate falls back to Torch. Fusion is
off by default. [Operator support, fallback and fusion](docs/ascend.md) ·
[中文编译与执行流程](docs/ascend-backend.zh.md)

## Examples and verification

With pytest installed, run the functional examples on an NPU. The first
command runs the default-target cases, which use AscendC when
`TILELANG_DEFAULT_TARGET` is unset. The second selects the explicit PTO case.
Each example also runs as a standalone script and accepts `--device`, `--dtype` and `--seed`.
[Example programs](examples/ascend)

```bash
python -m pytest examples/ascend -q -m "not pto"
python -m pytest examples/ascend -q -m pto
```

For CPU-side unit tests, source-generation checks, hardware verification and
benchmark interpretation, see the [verification guide](docs/verification.md).

## Origin and license

Extracted from TileLang at `7eb3edc8ba7b183c63681aa149a2e7512057d5ec`.
[Upstream PR #3892](https://github.com/tile-ai/tilelang/pull/3892) · [MIT license](LICENSE)
