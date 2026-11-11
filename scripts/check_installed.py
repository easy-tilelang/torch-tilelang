"""Smoke-test an installed wheel from outside the project source directory."""

import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch._dynamo.backends.registry import lookup_backend


def main():
    assert "torch_tilelang" not in sys.modules
    backend = lookup_backend("torch_tilelang_ascend")
    assert backend.__module__ == "torch_tilelang.backend"

    import tilelang
    import torch_tilelang

    package = Path(torch_tilelang.__file__).resolve()
    # A wheel check must not silently resolve the source checkout's editable package.
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    assert importlib.util.find_spec("tilelang.graph") is None
    assert importlib.util.find_spec("tilelang.ascend.graph") is None

    def model(x):
        return torch.relu(x + 1)

    compiled = torch.compile(model, backend="torch_tilelang_ascend", fullgraph=True)
    x = torch.arange(-4.0, 4.0)
    torch.testing.assert_close(compiled(x), model(x))
    print(json.dumps({"package": str(package), "tilelang": tilelang.__file__, "backend": backend.__module__}))


if __name__ == "__main__":
    main()
