"""Compile a PyTorch function for Ascend and compare its output with eager execution."""

import torch


def model(x, bias):
    return torch.relu(x + bias) * bias


if __name__ == "__main__":
    x = torch.randn(7, 65, device="npu")
    bias = torch.randn(65, device="npu")
    compiled = torch.compile(model, backend="tilelang", fullgraph=True, options={"strict": True})
    torch.testing.assert_close(compiled(x, bias), model(x, bias))
    print("Compiled output matches eager.")
