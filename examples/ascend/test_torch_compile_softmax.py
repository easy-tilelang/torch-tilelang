"""Check last-axis Softmax through the installed torch.compile backend.

Run directly or with pytest; --device, --dtype and --seed are optional:
    python examples/ascend/test_torch_compile_softmax.py
"""

import sys

import pytest
import torch


pytestmark = pytest.mark.ascend
SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")


def test_torch_compile_softmax(device, dtype, seed):
    torch.manual_seed(seed)
    model = torch.nn.Softmax(dim=-1).eval().to(device=device, dtype=dtype)
    compiled = torch.compile(
        model,
        backend="tilelang",
        fullgraph=True,
        dynamic=False,
        options={"strict": True, "fusion": "off"},
    )
    rtol, atol = {
        torch.float16: (1e-2, 1e-2),
        torch.bfloat16: (2e-2, 2e-2),
        torch.float32: (1e-4, 1e-4),
    }[dtype]
    inputs = [
        torch.randn(7, 128, device=device, dtype=dtype) * 0.125,
        torch.zeros(7, 128, device=device, dtype=dtype),
        torch.randn(7, 128, device=device, dtype=dtype) * 2,
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
