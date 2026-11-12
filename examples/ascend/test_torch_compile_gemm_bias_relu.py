"""Check a parameterized 2-D GEMM + bias + ReLU module through torch.compile.

Run directly or with pytest; --device, --dtype and --seed are optional:
    python examples/ascend/test_torch_compile_gemm_bias_relu.py
"""

import sys

import pytest
import torch


pytestmark = pytest.mark.ascend
SUPPORTED_DTYPES = ("float16", "bfloat16")


class GemmBiasRelu(torch.nn.Module):
    def __init__(self, inner, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(inner, width) * 0.125)
        self.bias = torch.nn.Parameter(torch.randn(width) * 0.125)

    def forward(self, x):
        return torch.relu(x @ self.weight + self.bias)


def test_torch_compile_gemm_bias_relu(device, dtype, seed):
    torch.manual_seed(seed)
    model = GemmBiasRelu(64, 64).eval().to(device=device, dtype=dtype)
    compiled = torch.compile(
        model,
        backend="torch_tilelang_ascend",
        fullgraph=True,
        dynamic=False,
        options={"strict": True, "fusion": "off"},
    )
    rtol, atol = {
        torch.float16: (1e-2, 1e-2),
        torch.bfloat16: (2e-2, 2e-2),
    }[dtype]
    inputs = [
        torch.randn(32, 64, device=device, dtype=dtype) * 0.125,
        torch.zeros(32, 64, device=device, dtype=dtype),
        torch.randn(32, 64, device=device, dtype=dtype) * 2,
    ]
    previous = []
    with torch.no_grad():
        for values in inputs:
            expected = model(values)
            actual = compiled(values)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
            for old_output, snapshot in previous:
                torch.testing.assert_close(old_output, snapshot, rtol=0, atol=0, equal_nan=True)
            previous.append((actual, actual.clone()))
        torch.npu.synchronize(device)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
