"""Experimental Torch-managed Ascend graph compilation.

Install this package to expose ``backend="torch_tilelang_ascend"`` through
Torch Dynamo's package entry-point discovery. Import this module for the
programmatic configuration and backend factory API.
"""

from .backend import make_backend, tilelang_backend
from .config import GraphConfig
from .frontend import UnsupportedGraphError, fx_to_relax
from .runtime import build_runner

__all__ = ["GraphConfig", "UnsupportedGraphError", "build_runner", "fx_to_relax", "make_backend", "tilelang_backend"]
