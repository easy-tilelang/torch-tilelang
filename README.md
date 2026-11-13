# torch-tilelang

A `torch.compile` backend powered by TileLang, with initial support for Ascend.

Repository: [easy-tilelang/torch-tilelang](https://github.com/easy-tilelang/torch-tilelang).

This experimental inference compiler converts supported FX partitions through
Relax and scalar TIR into Ascend TileLang kernels. Torch owns intermediate
tensors. Compilation, caching and launch use TileLang's existing AscendC/TVM-FFI
or PTO implementation.

```python
import torch

def model(x, bias):
    return torch.relu(x + bias) * bias

# Discovered through the installed torch_dynamo_backends entry point.
compiled = torch.compile(model, backend="torch_tilelang_ascend")
```

## Installation

Install a compatible Torch/torch_npu environment and an Ascend-enabled TileLang
build with its bundled TVM/Relax first. Then install this package:

```bash
python -m pip install --no-deps -e .
```

The package itself is Python-only. NPU execution additionally needs the CANN and
compiler environment required by TileLang's AscendC/TVM-FFI backend. The tested
TileLang revision and required interfaces are recorded in
[compatibility](docs/compatibility.md). Optional PTO execution is described in
[the Ascend guide](docs/ascend.md).

See [the Ascend guide](docs/ascend.md), [compatibility](docs/compatibility.md),
[extraction provenance](docs/migration.md), and [verification](docs/verification.md).

## Functional examples

With pytest installed in the compatible environment, run all four standalone
examples through the installed `torch_tilelang_ascend` backend:

```bash
python -m pytest examples/ascend -q
```

| Example | Input shape | Supported dtypes |
| --- | --- | --- |
| [Bias + ReLU + Scale](examples/ascend/test_torch_compile_elementwise.py) | `[7, 65]` | fp16, bf16, fp32 |
| [Softmax](examples/ascend/test_torch_compile_softmax.py) | `[7, 128]` | fp16, bf16, fp32 |
| [RMSNorm](examples/ascend/test_torch_compile_rmsnorm.py) | `[7, 128]` | fp16, bf16, fp32 |
| [GEMM + bias + ReLU](examples/ascend/test_torch_compile_gemm_bias_relu.py) | `[32, 64]`, weight `[64, 64]` | fp16, bf16 |

All options are optional: `--device` defaults to `npu:0`, `--dtype` to `all`
supported dtypes per example, and `--seed` to `0`. Override them as needed:

```bash
python -m pytest examples/ascend -q --device npu:1 --dtype bfloat16 --seed 0
```

Each script can also run directly with the same optional arguments:

```bash
python examples/ascend/test_torch_compile_rmsnorm.py
python examples/ascend/test_torch_compile_rmsnorm.py --device npu:1 --dtype float32
```

The examples use strict compilation with fusion disabled. Each dtype checks
three different inputs against Torch and verifies that later calls preserve
previous outputs. A complete hardware run reports **11 passed**, covering 33
requests. Missing Ascend compiler support or NPU hardware produces explicit
skips; skips do not establish successful device execution. Selecting `float32`
skips the GEMM example because that schedule supports fp16/bf16 inputs.
The examples run separately from the default `tests/` collection.

## Development and verification

After installing pytest and ruff in the compatible environment:

```bash
python -m pytest tests -q -m "not pto"
python -m ruff check src tests examples scripts
python -m pip wheel . --no-deps --no-build-isolation
```

Unit tests inject CPU kernels while retaining the real FX/Relax frontend and
graph executor. Source-only tests require an Ascend-enabled TileLang build.
Runtime tests additionally require an NPU; PTO cases have the `pto` marker.
Run optional PTO tests with `python -m pytest tests/ascend -q -m pto` against a
separately validated PTO environment; the tested compiler baseline has a known
PTO rectangular-GEMM codegen failure recorded in [verification](docs/verification.md).
Runtime, non-default-stream and fresh-process cache checks must pass on actual
hardware before publishing a hardware compatibility claim.

Run the complete AscendC hardware acceptance workflow with defaults:

```bash
python scripts/verify_npu.py
```

This selects the first idle, healthy NPU inside the visible allocation and
creates a unique result directory beside the checkout under
`torch-tilelang-artifacts/`. It runs the regression suite, functional examples,
a 30-minute soak and three-process performance measurements. Override the
device or output with `--physical-device` or `--output`; add `--include-pto`
for the independent PTO audit. See
[verification](docs/verification.md) for stage selection, recorded NPU results
and the remaining compatibility/performance blockers.

## Scope

Initial compiled inputs are static, contiguous, zero-offset fp16/bf16/fp32
tensors on one NPU. Supported families are injective expressions, last-axis
reductions and aligned 2-D fp16/bf16 GEMM. Training, dynamic or general strided
layouts, cross-core reductions and general Vector-to-Cube/Cube-to-Cube fusion
remain outside the capability contract.

Unsupported pure operators stay in Torch. Unsafe effects retain the original
graph. Numerical validation can reject a partition; device execution errors
propagate. Fusion is off by default; `options={"fusion": "profile"}` enables
selection using real correctness and timing evidence.

## Origin and license

Extracted from [tile-ai/tilelang PR #3892](https://github.com/tile-ai/tilelang/pull/3892)
at `7eb3edc8ba7b183c63681aa149a2e7512057d5ec`. Original Tile-AI license text
is preserved in [LICENSE](LICENSE).
