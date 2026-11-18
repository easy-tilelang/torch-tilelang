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
compiled = torch.compile(model, backend="tilelang")
```

Select AscendC or PTO with `options={"target": "ascendc"}` or
`options={"target": "pto"}`. An explicit target takes precedence over
`TILELANG_DEFAULT_TARGET`. With neither set, the backend uses AscendC for
`dav-3510`. CUDA is currently unsupported. The former backend name
`torch_tilelang_ascend` remains available as an alias.

## Installation

Install a compatible Torch/torch_npu environment and an Ascend-enabled TileLang
build with its bundled TVM/Relax first. Then install this package:

```bash
python -m pip install --no-deps -e .
```

Rerun this command when upgrading an editable checkout to register the new
`tilelang` backend name, then start a fresh Python process.

The package itself is Python-only. NPU execution also requires CANN and the
compiler toolchain for the selected backend. See the
[compatibility guide](docs/compatibility.md) for compiler revisions and
dependencies, and the [Ascend guide](docs/ascend.md) for backend options.

## Functional examples

With pytest installed, run the [torch.compile example](examples/ascend/test_torch_compile.py)
on an Ascend NPU. It checks the default target and an explicit PTO target.

```bash
python examples/ascend/test_torch_compile.py
python examples/ascend/test_torch_compile.py -m pto
```

Run all examples, or select the cases that do not require PTO with `-m "not pto"`.

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

Each functional test script also accepts these options when run directly.

```bash
python examples/ascend/test_torch_compile_rmsnorm.py
python examples/ascend/test_torch_compile_rmsnorm.py --device npu:1 --dtype float32
TILELANG_DEFAULT_TARGET=pto python examples/ascend/test_torch_compile_rmsnorm.py
```

The four operator tests listed above use strict compilation with fusion disabled. For each supported
dtype, they compare three inputs against Torch and check that later calls
preserve earlier outputs. Pytest skips cases when the required Ascend compiler,
NPU or dtype is unavailable. Run the examples separately from the default
`tests/` collection.

## Development and verification

After installing pytest and ruff in the compatible environment:

```bash
python -m pytest tests -q -m "not pto"
python -m ruff check src tests examples scripts
python -m pip wheel . --no-deps --no-build-isolation
```

Unit tests can inject CPU kernels while exercising the FX/Relax frontend and
graph executor. Source generation requires an Ascend-enabled TileLang build.
Device execution, stream handling and PTO disk-cache restoration require an NPU.

Run the complete AscendC hardware acceptance workflow with defaults:

```bash
python scripts/verify_npu.py
```

The driver selects an idle, healthy NPU in the visible allocation and writes
results outside the checkout. Run the PTO workflow separately with

```bash
python scripts/verify_npu.py --stage pto
```

The [verification guide](docs/verification.md) explains stage selection, soak
acceptance criteria and how to interpret benchmark comparisons.

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
at `7eb3edc8ba7b183c63681aa149a2e7512057d5ec`. The
[migration notes](docs/migration.md) map the original files to this package.
Original Tile-AI license text is preserved in [LICENSE](LICENSE).
