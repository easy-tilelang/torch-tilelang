"""Resolve existing TileLang targets and use its shared compiler/cache."""

from dataclasses import replace
import json

import tilelang
from tilelang import tvm
from tvm.target import Target

from ..config import DEFAULT_ASCEND_TARGET, GraphConfig
from ..frontend import UnsupportedGraphError


def resolve_graph_context(config: GraphConfig):
    """Validate the existing AscendC/PTO pair without probing the NPU."""
    from tilelang.backend import create_backend_context

    target = config.target.strip()
    kind = json.loads(target).get("kind") if target.startswith("{") else target.split()[0]
    if kind not in ("ascend", "asc", "ascendc", "pto", "auto"):
        raise UnsupportedGraphError(f"Unsupported TileLang graph target {kind!r}; supported targets are 'ascendc' and 'pto'")
    if target == "ascendc":
        if config.execution_backend == "pto":
            raise ValueError("The 'ascendc' target requires 'tvm_ffi' or 'auto' execution")
        target = DEFAULT_ASCEND_TARGET
    if "ascend" not in Target.list_kinds() or tvm.get_global_func("tl.TargetIsAscend", allow_missing=True) is None:
        raise UnsupportedGraphError("Ascend graph compilation is unavailable in this TileLang build; rebuild with USE_ASCEND=ON")
    context = create_backend_context(target, execution_backend=config.execution_backend)
    if context.module.name == "pto":
        if config.execution_backend == "tvm_ffi":
            raise ValueError("Invalid execution backend 'tvm_ffi' for a PTO graph target; use 'pto' or 'auto'")
        if config.execution_backend == "auto" and context.execution_backend.name != "pto":
            # TileLang may prefer TVM-FFI for this codegen variant. Graph
            # execution requires its PTO adapter; resolve that pair explicitly
            # without changing TileLang's process-wide backend registry.
            context = create_backend_context(target, execution_backend="pto")
    codegen = {
        ("ascend", "tvm_ffi"): "target.build.tilelang_ascend",
        ("pto", "pto"): "target.build.tilelang_pto_without_compile",
    }.get((context.module.name, context.execution_backend.name))
    if context.target.kind.name != "ascend" or codegen is None:
        raise UnsupportedGraphError("The graph backend supports AscendC with tvm_ffi or PTO with pto execution")
    if tvm.get_global_func(codegen, allow_missing=True) is None:
        raise UnsupportedGraphError(
            f"Graph backend {context.module.name!r} requires native codegen {codegen!r}; rebuild TileLang with USE_ASCEND=ON"
        )
    return context


def resolved_config(config: GraphConfig, context) -> GraphConfig:
    """Freeze a preflight choice into the graph's immutable options."""
    return replace(config, target=str(context.target), execution_backend=context.execution_backend.name)


def compile_kernel(func, config: GraphConfig):
    """Compile a scheduled PrimFunc with caller-owned output buffers."""
    return tilelang.compile(func, target=config.target, execution_backend=config.execution_backend, out_idx=[])


def lower_for_source(func, config: GraphConfig):
    """Generate source using the same target variant as graph execution."""
    context = resolve_graph_context(config)
    with context.target:
        return tilelang.lower(func, target=context.target)
