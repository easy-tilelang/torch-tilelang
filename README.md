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

## Development and verification

After installing pytest and ruff in the compatible environment:

```bash
python -m pytest tests -q -m "not pto"
python -m ruff check src tests examples
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
