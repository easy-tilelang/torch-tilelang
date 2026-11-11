"""The installed entry point works before importing the external package."""

import subprocess
import sys
import textwrap


def test_backend_entry_point_in_a_fresh_process(tmp_path):
    probe = textwrap.dedent(
        """
        import sys
        import torch
        from torch._dynamo.backends.registry import lookup_backend

        assert 'torch_tilelang' not in sys.modules
        backend = lookup_backend('torch_tilelang_ascend')
        assert backend.__module__ == 'torch_tilelang.backend'
        compiled = torch.compile(lambda x: torch.relu(x + 1), backend='torch_tilelang_ascend', fullgraph=True)
        x = torch.arange(-4.0, 4.0)
        torch.testing.assert_close(compiled(x), torch.relu(x + 1))
        assert not any(name == 'tilelang.graph' or name.startswith('tilelang.graph.') for name in sys.modules)
        assert not any(name == 'tilelang.ascend.graph' or name.startswith('tilelang.ascend.graph.') for name in sys.modules)
        print('external backend loaded')
        """
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "external backend loaded" in result.stdout
