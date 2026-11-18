"""The installed entry point works before importing the external package."""

import subprocess
import sys
import textwrap

import pytest
import torch


def test_backend_entry_point_in_a_fresh_process(tmp_path):
    probe = textwrap.dedent(
        """
        import sys
        import torch
        from torch._dynamo.backends.registry import lookup_backend

        assert 'torch_tilelang' not in sys.modules
        def model(x):
            return torch.relu(x + 1)

        x = torch.arange(-4.0, 4.0)
        for name in ('tilelang', 'torch_tilelang_ascend'):
            assert lookup_backend(name).__module__ == 'torch_tilelang.backend'
            compiled = torch.compile(model, backend=name, fullgraph=True)
            torch.testing.assert_close(compiled(x), model(x))
        assert not any(name == 'tilelang.graph' or name.startswith('tilelang.graph.') for name in sys.modules)
        assert not any(name == 'tilelang.ascend.graph' or name.startswith('tilelang.ascend.graph.') for name in sys.modules)
        print('external backend loaded')
        """
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "external backend loaded" in result.stdout


@pytest.mark.parametrize("env_target,options", [("cuda", {}), ("ascendc", {"target": "cuda"})])
def test_public_backend_rejects_cuda(monkeypatch, env_target, options):
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", env_target)
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: x + 1, backend="tilelang", fullgraph=True, options=options)
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed, match=r"(?i)cuda"):
        compiled(torch.ones(4))
