"""Source-only tests need Ascend codegen; runtime tests have additional markers."""

import tilelang.testing

if not tilelang.testing.ascend_backend_compiled():
    collect_ignore_glob = ["test_*.py"]
