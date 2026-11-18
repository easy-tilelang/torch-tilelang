"""Check the default target and an explicit PTO target against PyTorch eager."""

import sys

import pytest
import torch


pytestmark = pytest.mark.ascend
SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")


def model(x, bias):
    return torch.relu(x + bias) * bias


@pytest.mark.parametrize("options", [
    pytest.param({"strict": True}, id="default"),
    pytest.param({"strict": True, "target": "pto"}, id="pto", marks=pytest.mark.pto),
])
def test_torch_compile(device, dtype, seed, options):
    torch.manual_seed(seed)
    x = torch.randn(7, 65, device=device, dtype=dtype)
    bias = torch.randn(65, device=device, dtype=dtype)
    compiled = torch.compile(model, backend="tilelang", fullgraph=True, options=options)
    torch.testing.assert_close(compiled(x, bias), model(x, bias))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
